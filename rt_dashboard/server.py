from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .inference import SERVICE, metrics_payload


ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "rt_dashboard" / "web"


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "SMRTDashboard/1.0"

    def log_message(self, fmt: str, *args) -> None:
        print(f"[SMRT] {self.address_string()} - {fmt % args}")

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/status":
            self._json(SERVICE.status())
            return
        if path == "/api/metrics":
            self._json(metrics_payload())
            return
        if path == "/api/examples":
            self._json(
                {
                    "examples": [
                        {
                            "name": "test_007438",
                            "smiles": "O=c1n(CCOc2ccc(F)cc2)nc2ccc(S(=O)(=O)N3CCCC3)cn12",
                        },
                        {
                            "name": "test_002941",
                            "smiles": "CC(C)CN1C(=O)c2cccn2C[C@H]1C(O)=NCCN1CCOCC1",
                        },
                        {
                            "name": "test_005910",
                            "smiles": "Cc1cc(OCC(O)=NCc2ccc(Cl)cc2)nc(N2CCN(c3ccccc3)CC2)n1",
                        },
                    ]
                }
            )
            return

        relative = "index.html" if path in {"", "/"} else path.lstrip("/")
        target = (WEB_ROOT / relative).resolve()
        if WEB_ROOT.resolve() not in target.parents and target != WEB_ROOT.resolve():
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = target.read_bytes()
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type + ("; charset=utf-8" if content_type.startswith("text/") else ""))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/predict":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            result = SERVICE.predict(body.get("smiles", ""))
            self._json({"ok": True, "result": result})
        except ValueError as exc:
            self._json({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            self._json({"ok": False, "error": f"模型推理失败：{type(exc).__name__}: {exc}"}, status=500)


def main() -> None:
    parser = argparse.ArgumentParser(description="SMRT retention-time prediction dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true")
    parser.add_argument("--warmup", action="store_true", help="启动时预加载模型")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    if args.warmup:
        threading.Thread(target=SERVICE._load, daemon=True).start()
    if args.open:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"SMRT dashboard: {url}")
    print("首次预测会加载模型，后续预测会明显加快。按 Ctrl+C 停止。")
    ThreadingHTTPServer((args.host, args.port), DashboardHandler).serve_forever()


if __name__ == "__main__":
    main()
