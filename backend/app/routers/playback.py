"""Same-origin HLS playback using only fixed internal media paths."""

import re
from typing import Annotated
from urllib.parse import parse_qsl, urlsplit
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models import Camera
from app.repositories.assets import CameraRepository

router = APIRouter(prefix="/api/v1/cameras", tags=["playback"])
Session = Annotated[AsyncSession, Depends(get_db, scope="function")]
ASSET = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\.(?:m3u8|mp4|ts)")
MAX_PLAYLIST_BYTES = 262144
MAX_MEDIA_BYTES = 16777216


def unavailable() -> HTTPException:
    return HTTPException(503, "Camera playback is unavailable")


async def active_camera(identifier: UUID, session: AsyncSession) -> Camera:
    camera = await CameraRepository(session).get(identifier)
    if not camera.is_active:
        raise HTTPException(409, "Camera is inactive")
    return camera


def safe_asset(value: str) -> bool:
    return bool(ASSET.fullmatch(value)) and ".." not in value


def safe_query(items: list[tuple[str, str]]) -> dict[str, str]:
    result = {}
    for name, value in items:
        if name in result:
            raise HTTPException(422, "Invalid playback query")
        if name in {"_HLS_msn", "_HLS_part"}:
            if not value.isascii() or not value.isdecimal() or len(value) > 15:
                raise HTTPException(422, "Invalid playback query")
        elif name == "_HLS_skip":
            if value not in {"YES", "v2"}:
                raise HTTPException(422, "Invalid playback query")
        else:
            raise HTTPException(422, "Invalid playback query")
        result[name] = value
    return result


def validate_playlist(content: bytes) -> bytes:
    try:
        text = content.decode("utf-8", errors="strict")
        if not text.startswith("#EXTM3U") or "://" in text or "\\" in text:
            raise ValueError
        for line in text.splitlines():
            references = re.findall(r'URI="([^"]+)"', line) if line.startswith("#") else [line]
            for value in references:
                if not value:
                    continue
                parsed = urlsplit(value)
                if parsed.scheme or parsed.netloc or parsed.fragment or not safe_asset(parsed.path):
                    raise ValueError
                safe_query(parse_qsl(parsed.query, keep_blank_values=True))
        return content
    except (UnicodeError, ValueError, HTTPException):
        raise unavailable() from None


@router.get("/{identifier}/playback")
async def playback_descriptor(identifier: UUID, session: Session) -> dict:
    await active_camera(identifier, session)
    base = f"/api/v1/cameras/{identifier}"
    return {
        "camera_id": str(identifier),
        "hls_url": f"{base}/hls/index.m3u8",
        "player_url": f"{base}/player",
        "status": "CONFIGURED",
        "content": "SOURCE_VIDEO",
    }


@router.get("/{identifier}/hls/{asset}")
async def playback_asset(
    identifier: UUID, asset: str, request: Request, session: Session
) -> Response:
    await active_camera(identifier, session)
    if not safe_asset(asset):
        raise HTTPException(422, "Invalid playback asset")
    query = safe_query(list(request.query_params.multi_items()))
    settings = request.app.state.settings
    maximum = MAX_PLAYLIST_BYTES if asset.endswith(".m3u8") else MAX_MEDIA_BYTES
    try:
        async with httpx.AsyncClient(
            base_url=settings.mediamtx_hls_url,
            follow_redirects=False,
            trust_env=False,
            timeout=settings.dependency_timeout_seconds,
            headers={
                "Accept-Encoding": "identity",
                "Authorization": f"Bearer {settings.mediamtx_read_password.get_secret_value()}",
            },
        ) as client:
            async with client.stream(
                "GET", f"/cameras/{identifier}/{asset}", params=query
            ) as upstream:
                if (
                    upstream.status_code != 200
                    or upstream.headers.get("content-encoding", "identity") != "identity"
                ):
                    raise unavailable()
                content = bytearray()
                async for chunk in upstream.aiter_bytes(chunk_size=65536):
                    if len(content) + len(chunk) > maximum:
                        raise unavailable()
                    content.extend(chunk)
        if asset.endswith(".m3u8"):
            body = validate_playlist(bytes(content))
            media_type = "application/vnd.apple.mpegurl"
        else:
            body = bytes(content)
            media_type = "video/mp2t" if asset.endswith(".ts") else "video/mp4"
        return Response(body, media_type=media_type, headers={"Cache-Control": "private, no-store"})
    except httpx.HTTPError:
        raise unavailable() from None


@router.get("/{identifier}/player", response_class=Response)
async def playback_player(identifier: UUID, session: Session) -> Response:
    await active_camera(identifier, session)
    source = f"/api/v1/cameras/{identifier}/hls/index.m3u8"
    page = f'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Camera Source Playback</title>
<style>
body{{font:16px system-ui;background:#111827;color:#f9fafb;
max-width:1280px;margin:2rem auto;padding:1rem}}
video{{width:100%;background:#000}}a{{color:#7dd3fc}}
</style>
<h1>Camera Source Playback</h1>
<p>This is the original source video.
Worker inference and annotated diagnostic clips are separate outputs.</p>
<video id="video" controls autoplay muted playsinline></video>
<p id="status">Connecting to source video.</p>
<p><a href="{source}">Open the HLS playlist in a compatible player</a></p>
<script src="/static/hls.min.js"></script><script>
const video=document.getElementById('video');
const status=document.getElementById('status');
const source={source!r};
if(video.canPlayType('application/vnd.apple.mpegurl')){{video.src=source;}}
else if(window.Hls && Hls.isSupported()){{
  const hls=new Hls({{lowLatencyMode:true}});
  hls.loadSource(source);hls.attachMedia(video);
  hls.on(Hls.Events.ERROR,(_,data)=>{{
    if(data.fatal)status.textContent='Source unavailable. Retry after the camera reconnects.';
  }});
}}
else{{
  status.textContent='This browser needs HLS support. Open the playlist in a compatible player.';
}}
video.addEventListener('playing',()=>{{status.textContent='Playing source video.';}});
video.addEventListener('error',()=>{{
  status.textContent='Source unavailable or unsupported codec.';
}});
</script></html>'''
    return Response(
        page,
        media_type="text/html",
        headers={
            "Cache-Control": "private, no-store",
            "Content-Security-Policy": (
                "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                "style-src 'unsafe-inline'; media-src 'self' blob:; connect-src 'self'; "
                "worker-src blob:; object-src 'none'; base-uri 'none'"
            ),
        },
    )
