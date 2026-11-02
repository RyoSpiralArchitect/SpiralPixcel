"""USB-local drawing session. Only a submitted drawing leaves the page."""
import argparse
import base64
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import subprocess
import tarfile
import threading
import uuid

from touch import decode, MAX_BYTES


def serve(adb, connect, execute):
    page = Path(__file__).with_name("draw.html").read_bytes()
    prefix = "/" + secrets.token_urlsafe(24) + "/"
    busy = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # The session URL stays out of access logs.

        def send(self, status, content, kind):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            if self.path == prefix:
                self.send(200, page, "text/html; charset=utf-8")
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path != prefix + "trace":
                return self.send_error(404)
            if self.headers.get("Origin") not in origins:
                return self.send_error(403)
            if self.headers.get_content_type() != "application/json":
                return self.send_error(415)
            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self.send_error(400)
            if not 0 < size <= MAX_BYTES:
                return self.send_error(413)
            if not busy.acquire(blocking=False):
                return self.send(409, b"comparison already running", "text/plain")
            try:
                self.connection.settimeout(20)
                raw = self.rfile.read(size)
                decode(raw)
                name = datetime.now(timezone.utc).strftime("touch-%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
                dest = Path("runs") / name
                dest.mkdir(parents=True)
                trace = dest / "trace.json"
                trace.write_bytes(raw)
                print(f"Saved {trace.resolve()}", flush=True)
                ssh = connect()
                images, ids = [], []
                for rate in (0.08, 0.18):
                    args = argparse.Namespace(command="run", steps=24, decay=0.995, rate=rate, trace=trace)
                    run = execute(ssh, args)
                    ids.append(run.name)
                    (dest / "comparison.json").write_text(json.dumps({"runs": ids, "trace": trace.name}) + "\n")
                    png = base64.b64encode((run / "field.png").read_bytes()).decode("ascii")
                    images.append({"rate": rate, "png": png})
                body = json.dumps({"images": images, "capture": name}).encode()
            except (ValueError, OSError, subprocess.SubprocessError, tarfile.TarError) as exc:
                print(f"Touch comparison: {exc}", flush=True)
                self.send(422, "記録または計算に失敗しました。Macのログを確認してください。".encode(), "text/plain; charset=utf-8")
            else:
                self.send(200, body, "application/json")
            finally:
                busy.release()

    class Server(ThreadingHTTPServer):
        # Allow a submitted capture to finish writing its receipts if Ctrl-C closes the studio.
        daemon_threads = False

        def get_request(self):
            connection, address = super().get_request()
            connection.settimeout(15)  # Browsers may open speculative, idle connections.
            return connection, address

    with Server(("127.0.0.1", 0), Handler) as server:
        port = server.server_port
        remote = int(adb("reverse", "tcp:0", f"tcp:{port}").decode().strip())
        phone = f"http://127.0.0.1:{remote}"
        local = f"http://127.0.0.1:{port}"
        origins = {local, phone}
        try:
            adb("shell", "am", "start", "-a", "android.intent.action.VIEW", "-d", phone + prefix)
            print(f"Pixel: {phone}{prefix}\nMac:   {local}{prefix}\nCtrl-C to close.", flush=True)
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            try:
                adb("reverse", "--remove", f"tcp:{remote}")
            except subprocess.SubprocessError:
                pass
