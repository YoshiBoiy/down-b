from __future__ import annotations

import argparse
import json
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote


UPLOAD_SUBDIR = Path("uploads/avatar_models")
BUILTIN_MODEL_SUBDIR = Path("../models")
ALLOWED_MODEL_EXTENSIONS = {".glb", ".gltf"}


class CacheSafeHandler(SimpleHTTPRequestHandler):
    uploads_dir: Path

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self) -> None:
        if self.path == "/api/avatars":
            self._send_avatar_list()
            return
        super().do_GET()

    def do_POST(self) -> None:
        if self.path == "/api/upload-avatar":
            self._handle_avatar_upload()
            return
        self.send_error(404, "Not found")

    def _send_avatar_list(self) -> None:
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        builtins_dir = (Path(self.directory).resolve() / BUILTIN_MODEL_SUBDIR).resolve()
        builtin_paths = sorted(builtins_dir.iterdir()) if builtins_dir.exists() else []
        builtins = [
            {
                "name": path.stem,
                "url": f"/models/{quote(path.name)}",
                "filename": path.name,
                "source": "models",
            }
            for path in builtin_paths
            if path.is_file() and path.suffix.lower() in ALLOWED_MODEL_EXTENSIONS
        ]
        avatars = [
            {
                "name": path.stem,
                "url": f"/uploads/avatar_models/{quote(path.name)}",
                "filename": path.name,
                "source": "uploads",
            }
            for path in sorted(self.uploads_dir.iterdir())
            if path.is_file() and path.suffix.lower() in ALLOWED_MODEL_EXTENSIONS
        ]
        payload = json.dumps({"avatars": builtins + avatars}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _handle_avatar_upload(self) -> None:
        content_type = self.headers.get("Content-Type", "")
        boundary_marker = "boundary="
        if boundary_marker not in content_type:
            self.send_error(400, "Expected multipart/form-data")
            return
        boundary = content_type.split(boundary_marker, 1)[1].strip().strip('"').encode("utf-8")
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        file_name, file_bytes = _extract_first_file(body, boundary)
        if not file_name or file_bytes is None:
            self.send_error(400, "No model file found")
            return
        safe_name = _safe_filename(file_name)
        if Path(safe_name).suffix.lower() not in ALLOWED_MODEL_EXTENSIONS:
            self.send_error(400, "Only .glb and .gltf files are supported")
            return
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        target = self.uploads_dir / safe_name
        stem = target.stem
        suffix = target.suffix
        counter = 1
        while target.exists():
            target = self.uploads_dir / f"{stem}-{counter}{suffix}"
            counter += 1
        target.write_bytes(file_bytes)
        payload = json.dumps(
            {
                "name": target.stem,
                "filename": target.name,
                "url": f"/uploads/avatar_models/{quote(target.name)}",
            }
        ).encode("utf-8")
        self.send_response(201)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def translate_path(self, path: str) -> str:
        if path.startswith("/models/"):
            name = Path(path.removeprefix("/models/")).name
            return str((Path(self.directory).resolve() / BUILTIN_MODEL_SUBDIR / name).resolve())
        return super().translate_path(path)


def _extract_first_file(body: bytes, boundary: bytes) -> tuple[str | None, bytes | None]:
    delimiter = b"--" + boundary
    for part in body.split(delimiter):
        part = part.strip(b"\r\n")
        if not part or part == b"--" or b"\r\n\r\n" not in part:
            continue
        headers_blob, content = part.split(b"\r\n\r\n", 1)
        headers = headers_blob.decode("utf-8", errors="replace")
        if "filename=" not in headers:
            continue
        match = re.search(r'filename="([^"]+)"|filename=([^;\r\n]+)', headers)
        if not match:
            continue
        filename = (match.group(1) or match.group(2)).strip()
        return filename, content.rstrip(b"\r\n")
    return None, None


def _safe_filename(name: str) -> str:
    cleaned = Path(name).name.replace(" ", "_")
    return "".join(ch for ch in cleaned if ch.isalnum() or ch in {"-", "_", "."}) or "avatar.glb"


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Down-B laptop webcam app.")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--web-root", default="web")
    args = parser.parse_args()

    root = Path(args.web_root).resolve()
    uploads_dir = root / UPLOAD_SUBDIR
    uploads_dir.mkdir(parents=True, exist_ok=True)
    CacheSafeHandler.uploads_dir = uploads_dir
    handler = partial(CacheSafeHandler, directory=str(root))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Down-B webcam app running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
