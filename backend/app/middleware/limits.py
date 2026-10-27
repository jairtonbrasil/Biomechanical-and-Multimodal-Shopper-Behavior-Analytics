"""Bounded request bodies and shared per-client quotas for operational endpoints."""

import asyncio
import hashlib
import json

from starlette.types import ASGIApp, Message, Receive, Scope, Send

RATE_LIMIT = """
local count = redis.call('INCR', KEYS[1])
if count == 1 then redis.call('EXPIRE', KEYS[1], 60) end
return count
"""


class OperationalLimits:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    @staticmethod
    async def reject(send: Send, status: int, detail: str) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"cache-control", b"no-store"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": json.dumps({"detail": detail}).encode()})

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not scope["path"].startswith("/api/v1/"):
            await self.app(scope, receive, send)
            return
        settings = scope["app"].state.settings
        body = bytearray()
        try:
            async with asyncio.timeout(settings.dependency_timeout_seconds):
                while True:
                    event = await receive()
                    if event["type"] == "http.disconnect":
                        return
                    body.extend(event.get("body", b""))
                    if len(body) > settings.api_request_max_bytes:
                        await self.reject(send, 413, "Request body exceeds its size limit")
                        return
                    if not event.get("more_body", False):
                        break
        except TimeoutError:
            await self.reject(send, 408, "Request body deadline exceeded")
            return
        media_request = "/hls/" in scope["path"]
        quota = (
            settings.media_rate_limit_per_minute
            if media_request
            else settings.api_rate_limit_per_minute
        )
        if quota and settings.app_environment != "test":
            peer = (scope.get("client") or ("unknown", 0))[0]
            identifier = hashlib.sha256(peer.encode()).hexdigest()
            try:
                redis = scope["app"].state.resources.redis
                count = await redis.client.eval(
                    RATE_LIMIT,
                    1,
                    redis.key(
                        "channel", f"{'media-rate' if media_request else 'rate'}:{identifier}"
                    ),
                )
            except Exception:
                await self.reject(send, 503, "Request quota service is unavailable")
                return
            if count > quota:
                await self.reject(send, 429, "Request quota exceeded")
                return
        consumed = False

        async def bounded_receive() -> Message:
            nonlocal consumed
            if not consumed:
                consumed = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        async def secure_send(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                names = {name.lower() for name, _ in headers}
                if b"x-content-type-options" not in names:
                    headers.append((b"x-content-type-options", b"nosniff"))
                if b"cache-control" not in names:
                    headers.append((b"cache-control", b"no-store"))
                message["headers"] = headers
            await send(message)

        await self.app(scope, bounded_receive, secure_send)
