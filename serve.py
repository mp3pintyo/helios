"""Static file server with HTTP Range support, for streaming geometry in the browser.

`python -m http.server` ignores `Range` headers and always sends whole files (Helios then falls
back to downloading each .hvm once). This server answers single-range requests with
206 Partial Content, so pages stream on demand like they would from a CDN.

    python serve.py [PORT] [DIRECTORY]        (defaults: 8080, web)
"""

import os
import re
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

RANGE = re.compile(r"bytes=(\d*)-(\d*)$")


class RangeRequestHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        match = RANGE.match(self.headers.get("Range", "").strip())
        path = self.translate_path(self.path)
        if not match or not os.path.isfile(path):
            return super().send_head()
        size = os.path.getsize(path)
        first, last = match.groups()
        if first:
            start, end = int(first), min(int(last), size - 1) if last else size - 1
        elif last:
            start, end = max(size - int(last), 0), size - 1
        else:
            return super().send_head()
        if start > end or start >= size:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None
        f = open(path, "rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self.remaining = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        remaining = getattr(self, "remaining", None)
        if remaining is None:
            return super().copyfile(source, outputfile)
        while remaining > 0:
            chunk = source.read(min(remaining, 1 << 20))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)
        self.remaining = None


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    directory = sys.argv[2] if len(sys.argv) > 2 else "web"
    handler = partial(RangeRequestHandler, directory=directory)
    print(f"serving {os.path.abspath(directory)} on http://localhost:{port} (with Range support)")
    # A page streaming geometry opens dozens of range requests at once; the default listen
    # backlog (5) made Windows refuse some of them (ERR_CONNECTION_REFUSED).
    ThreadingHTTPServer.request_queue_size = 128
    ThreadingHTTPServer(("", port), handler).serve_forever()
