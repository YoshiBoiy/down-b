from __future__ import annotations

import asyncio
import json
import mimetypes
from pathlib import Path
from typing import Any

from websockets.asyncio.server import ServerConnection, serve
from websockets.datastructures import Headers
from websockets.http11 import Response


class WebSocketHub:
    def __init__(self):
        self.clients: set[ServerConnection] = set()

    async def handler(self, websocket: ServerConnection) -> None:
        self.clients.add(websocket)
        try:
            async for _ in websocket:
                pass
        finally:
            self.clients.discard(websocket)

    async def publish(self, message: dict[str, Any]) -> None:
        if not self.clients:
            return
        payload = json.dumps(message)
        await asyncio.gather(*(client.send(payload) for client in tuple(self.clients)), return_exceptions=True)


async def _http_response(web_root: Path, path: str) -> tuple[int, list[tuple[str, str]], bytes]:
    safe_path = "index.html" if path in {"", "/"} else path.lstrip("/")
    target = (web_root / safe_path).resolve()
    if web_root.resolve() not in target.parents and target != web_root.resolve():
        return 403, [("Content-Type", "text/plain")], b"forbidden"
    if not target.exists() or not target.is_file():
        return 404, [("Content-Type", "text/plain")], b"not found"
    content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    return 200, [("Content-Type", content_type)], target.read_bytes()


async def run_server(hub: WebSocketHub, web_root: str | Path, host: str, port: int) -> None:
    root = Path(web_root)

    async def process_request(connection: ServerConnection, request: Any):
        if request.path == "/ws":
            return None
        status, headers, body = await _http_response(root, request.path)
        response_headers = Headers()
        for key, value in headers:
            response_headers[key] = value
        reason = "OK" if status == 200 else "Error"
        return Response(status, reason, response_headers, body)

    async with serve(hub.handler, host, port, process_request=process_request):
        await asyncio.Future()
