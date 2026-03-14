# backend/redirect.py
import requests
from urllib.parse import urlparse
import ssl, socket, hashlib
from typing import Dict, Any
from functools import lru_cache

REQUEST_TIMEOUT = 8.0
HEADERS = {"User-Agent": "PhishFree-Analyzer/1.0"}

def follow_redirects(url: str, max_hops: int = 8) -> Dict[str, Any]:
    """
    Follow HTTP(S) redirects safely (no JS). Returns hops list, final URL, and status code.
    """
    try:
        s = requests.Session()
        s.max_redirects = max_hops
        # do not send cookies or auth by default
        resp = s.get(url, headers=HEADERS, allow_redirects=True, timeout=REQUEST_TIMEOUT)
        hops = [h.url for h in resp.history] + [resp.url]
        return {"hops": hops, "final_url": resp.url, "status_code": resp.status_code}
    except requests.exceptions.TooManyRedirects:
        return {"error": "too_many_redirects"}
    except requests.exceptions.RequestException as e:
        return {"error": f"request_error: {str(e)}"}
    except Exception as e:
        return {"error": str(e)}

@lru_cache(maxsize=1024)
def get_cert_fingerprint_cached(host: str, port: int, scheme: str = "https") -> Dict[str, str]:
    """
    Cached cert fingerprint retrieval. Input host should be hostname string (no credentials).
    Returns dict with cert_fp (colon-separated hex) or error.
    """
    try:
        if scheme.lower() != "https":
            return {"cert_fp": None, "note": "non-https scheme"}

        ctx = ssl.create_default_context()
        with socket.create_connection((host, port), timeout=6) as sock:
            with ctx.wrap_socket(sock, server_hostname=host) as ssock:
                der = ssock.getpeercert(binary_form=True)
                sha256 = hashlib.sha256(der).hexdigest()
                fp = ":".join(sha256[i:i+2].upper() for i in range(0, len(sha256), 2))
                return {"cert_fp": fp}
    except Exception as e:
        return {"error": str(e)}

def get_cert_fingerprint(url: str) -> Dict[str, str]:
    """
    Public function taking a URL and returning the cert fingerprint (uses cache).
    """
    try:
        parsed = urlparse(url)
        host = parsed.hostname
        if host is None:
            return {"error": "no host in url"}
        scheme = parsed.scheme or "http"
        port = parsed.port or (443 if scheme == "https" else 80)
        if scheme != "https":
            return {"cert_fp": None, "note": "non-https scheme"}
        return get_cert_fingerprint_cached(host, port, scheme)
    except Exception as e:
        return {"error": str(e)}
