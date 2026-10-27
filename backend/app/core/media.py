"""Authenticated, fixed-origin MediaMTX control without exposing source credentials."""

from uuid import UUID

import httpx

from app.core.config import Settings
from app.core.stream_policy import StreamPolicy
from app.domain.configuration import proxy_path
from app.models.camera import Camera


class MediaUnavailableError(RuntimeError):
    pass


class MediaMTXControl:
    MAX_RESPONSE_BYTES = 1048576
    MAX_LIST_PAGES = 100
    PAGE_SIZE = 100

    def __init__(
        self, settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self.settings = settings
        self.client = httpx.AsyncClient(
            base_url=settings.mediamtx_api_url,
            auth=("media-admin", settings.mediamtx_api_password.get_secret_value()),
            timeout=settings.dependency_timeout_seconds,
            follow_redirects=False,
            trust_env=False,
            headers={"Accept-Encoding": "identity"},
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
            transport=transport,
        )

    async def request(self, method: str, path: str, **kwargs) -> httpx.Response:
        try:
            async with self.client.stream(method, path, **kwargs) as streamed:
                if streamed.status_code not in (200, 404):
                    raise MediaUnavailableError("Media control rejected configuration")
                if streamed.headers.get("Content-Encoding", "identity").lower() != "identity":
                    raise MediaUnavailableError("Media control returned an unsupported encoding")
                content = bytearray()
                async for chunk in streamed.aiter_bytes(chunk_size=65536):
                    if len(content) + len(chunk) > self.MAX_RESPONSE_BYTES:
                        raise MediaUnavailableError("Media control response exceeds its size limit")
                    content.extend(chunk)
                return httpx.Response(streamed.status_code, content=bytes(content))
        except httpx.HTTPError:
            raise MediaUnavailableError("Media control is unavailable") from None

    @staticmethod
    def document(response: httpx.Response) -> dict:
        try:
            result = response.json()
        except (ValueError, RecursionError):
            raise MediaUnavailableError("Media control returned an invalid response") from None
        if response.status_code != 200 or not isinstance(result, dict):
            raise MediaUnavailableError("Media control returned an invalid response")
        return result

    async def apply(self, camera: Camera) -> None:
        path = proxy_path(camera.id)
        if not camera.is_active:
            await self.request("DELETE", f"/v3/config/paths/delete/{path}")
            return
        try:
            StreamPolicy(self.settings).validate(camera.stream_url)
        except ValueError:
            # A narrowed deployment allowlist must disable an already configured source.
            await self.request("DELETE", f"/v3/config/paths/delete/{path}")
            raise MediaUnavailableError("Stream destination is not allowed") from None
        desired = {"source": camera.stream_url, "sourceOnDemand": False, "rtspTransport": "tcp"}
        existing = await self.request("GET", f"/v3/config/paths/get/{path}")
        if existing.status_code == 404:
            result = await self.request("POST", f"/v3/config/paths/add/{path}", json=desired)
        else:
            current = self.document(existing)
            if all(current.get(key) == value for key, value in desired.items()):
                return
            result = await self.request("PATCH", f"/v3/config/paths/patch/{path}", json=desired)
        if result.status_code == 404:
            raise MediaUnavailableError("Media configuration changed concurrently")

    async def remove_orphans(self, camera_ids: set[UUID]) -> None:
        page = 0
        orphan_ids: set[UUID] = set()
        while True:
            response = await self.request(
                "GET",
                "/v3/config/paths/list",
                params={"page": page, "itemsPerPage": self.PAGE_SIZE},
            )
            document = self.document(response)
            items = document.get("items")
            page_count = document.get("pageCount")
            if (
                not isinstance(items, list)
                or len(items) > self.PAGE_SIZE
                or type(page_count) is not int
                or not 0 <= page_count <= self.MAX_LIST_PAGES
            ):
                raise MediaUnavailableError("Media control returned invalid pagination")
            for item in items:
                if (
                    not isinstance(item, dict)
                    or not isinstance(item.get("name"), str)
                    or not 1 <= len(item["name"]) <= 512
                ):
                    raise MediaUnavailableError("Media control returned an invalid path")
                name = item["name"]
                if name.startswith("cameras/"):
                    try:
                        identifier = UUID(name.removeprefix("cameras/"))
                    except ValueError:
                        continue
                    if identifier not in camera_ids:
                        orphan_ids.add(identifier)
            page += 1
            if page >= page_count:
                break
            if page >= self.MAX_LIST_PAGES:
                raise MediaUnavailableError("Media control pagination exceeds its limit")
        # Deleting while traversing offset pages could skip paths as later offsets shift.
        for identifier in sorted(orphan_ids):
            await self.request("DELETE", f"/v3/config/paths/delete/{proxy_path(identifier)}")

    async def close(self) -> None:
        await self.client.aclose()
