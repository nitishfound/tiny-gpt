"""Download Tiny Shakespeare (~1 MB, public domain) to data/shakespeare.txt."""

import ssl
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
OUT = Path(__file__).resolve().parents[1] / "data" / "shakespeare.txt"


def ssl_context() -> ssl.SSLContext:
    try:  # use the OS certificate store if available (works behind TLS-inspecting proxies)
        import truststore

        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except ImportError:
        return ssl.create_default_context()


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, context=ssl_context(), timeout=60) as resp:
        OUT.write_bytes(resp.read())
    print(f"saved {OUT.stat().st_size:,} bytes to {OUT}")
