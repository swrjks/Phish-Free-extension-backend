"""
url_analyzer.py  —  Standalone URL Phishing Heuristic Analyzer
================================================================
Usage:
    python url_analyzer.py
    > Paste a URL and press Enter.

No ML models. No page visits. Works 100% offline (except optional WHOIS).
All stdlib except `python-whois` (optional, for domain age).

Install optional dep:  pip install python-whois
"""

import re
import sys
import socket
from urllib.parse import urlparse, parse_qs
from datetime import datetime

# ── ANSI colours (work on Windows 10+ terminals) ─────────────────────────────
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

# ── Known-safe whitelist ──────────────────────────────────────────────────────
SAFE_DOMAINS = {
    "google.com", "www.google.com", "youtube.com", "www.youtube.com",
    "github.com", "www.github.com", "microsoft.com", "www.microsoft.com",
    "apple.com", "www.apple.com", "amazon.com", "www.amazon.com",
    "linkedin.com", "www.linkedin.com", "twitter.com", "x.com",
    "instagram.com", "facebook.com", "netflix.com", "wikipedia.org",
    "stackoverflow.com", "reddit.com", "whatsapp.com", "zoom.us",
}

# ── Risky TLDs ────────────────────────────────────────────────────────────────
RISKY_TLDS = {
    ".tk", ".ml", ".ga", ".cf", ".gq",          # Freenom free TLDs
    ".xyz", ".top", ".club", ".online", ".site", ".icu",
    ".ru", ".cn", ".ir", ".th",
}

# ── Suspicious keywords ───────────────────────────────────────────────────────
HOST_KEYWORDS = [
    "login", "signin", "secure", "verify", "account", "update",
    "banking", "paypal", "amazon", "apple", "google", "microsoft",
    "ebay", "confirm", "password", "credential", "wallet", "crypto",
    "invoice", "payment", "support", "helpdesk", "service", "webscr",
]

PATH_KEYWORDS = [
    "login", "signin", "verify", "secure", "account", "password",
    "update", "confirm", "redirect", "token", "credential", "reset",
    "session", "auth", "submit",
]

# ─────────────────────────────────────────────────────────────────────────────
def separator(char="─", width=60):
    print(DIM + char * width + RESET)

def section(title):
    print()
    print(BOLD + CYAN + f"  {title}" + RESET)
    separator()

def signal_row(name, value, flag=False, warn=False):
    icon  = (RED + "⚠ " if flag else (YELLOW + "• " if warn else GREEN + "✓ ")) + RESET
    val_c = RED + str(value) + RESET if flag else (YELLOW + str(value) + RESET if warn else str(value))
    print(f"    {icon} {name:<35} {val_c}")

# ─────────────────────────────────────────────────────────────────────────────
def analyze(raw_url: str):
    # Normalise
    url_to_parse = raw_url if re.match(r"^https?://", raw_url, re.I) else "http://" + raw_url
    try:
        parsed = urlparse(url_to_parse)
    except Exception as e:
        print(RED + f"  ✗ Invalid URL: {e}" + RESET)
        return

    scheme   = (parsed.scheme   or "").lower()
    host     = (parsed.hostname or "").lower()
    path     = (parsed.path     or "")
    query    = (parsed.query    or "")
    fragment = (parsed.fragment or "")
    port     = parsed.port
    full_url = raw_url

    parts        = host.split(".")
    tld          = ("." + parts[-1]) if parts else ""
    domain_name  = ".".join(parts[-2:]) if len(parts) >= 2 else host
    sub_count    = max(0, len(parts) - 2)

    score   = 0.0
    reasons = []
    signals = {}

    # ── Header ───────────────────────────────────────────────────────────────
    print()
    print(BOLD + "═" * 60 + RESET)
    print(BOLD + "  🔍  URL PHISHING ANALYZER" + RESET)
    print(BOLD + "═" * 60 + RESET)
    print(f"  {DIM}URL :{RESET} {full_url}")
    print(f"  {DIM}Host:{RESET} {host}")
    print(f"  {DIM}TLD :{RESET} {tld}")
    print(f"  {DIM}Time:{RESET} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── Whitelist check ───────────────────────────────────────────────────────
    if host in SAFE_DOMAINS:
        print()
        print(GREEN + BOLD + "  ✅  WHITELISTED DOMAIN — Considered Safe" + RESET)
        print(BOLD + "═" * 60 + RESET)
        return

    # ═════════════════════════════════════════════════════════════════════════
    # SIGNAL CHECKS
    # ═════════════════════════════════════════════════════════════════════════

    # 1. Scheme
    section("1. Protocol / Scheme")
    uses_http = scheme == "http"
    signals["uses_http"] = uses_http
    if uses_http:
        score += 0.10
        reasons.append("HTTP instead of HTTPS — unencrypted connection.")
    signal_row("Scheme", scheme.upper(), flag=uses_http)
    signal_row("HTTPS", "Yes" if scheme == "https" else "No", flag=uses_http)

    # 2. IP as host
    section("2. Host Analysis")
    ip_pat = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
    is_ip  = bool(ip_pat.match(host))
    signals["ip_as_host"] = is_ip
    if is_ip:
        score += 0.25
        reasons.append("Raw IP address used as host — phishing sites avoid real domains.")
    signal_row("Host",           host)
    signal_row("IP as Host",     is_ip, flag=is_ip)
    signal_row("Domain",         domain_name)
    signal_row("TLD",            tld, flag=tld in RISKY_TLDS, warn=(tld not in RISKY_TLDS and tld not in {".com",".org",".net",".edu",".gov"}))
    if tld in RISKY_TLDS:
        score += 0.15
        reasons.append(f"High-risk TLD '{tld}' — frequently abused in phishing.")
    signals["risky_tld"] = tld in RISKY_TLDS

    # Subdomains
    signal_row("Subdomains",     sub_count, flag=sub_count >= 3, warn=sub_count == 2)
    signals["subdomain_count"] = sub_count
    if sub_count >= 3:
        score += 0.15
        reasons.append(f"Deep subdomain nesting ({sub_count} levels) — classic phishing pattern.")
    elif sub_count == 2:
        score += 0.05
        reasons.append(f"Multiple subdomains ({sub_count} levels).")

    # 3. URL structure
    section("3. URL Structure")
    url_len = len(full_url)
    signals["url_length"] = url_len
    if url_len > 100:
        inc = min(0.15, (url_len - 100) / 400)
        score += inc
        reasons.append(f"Very long URL ({url_len} chars) — used to obscure the real destination.")
    signal_row("URL Length",          url_len, flag=url_len > 100, warn=url_len > 75)

    # @ symbol
    has_at = "@" in full_url
    signals["has_at_symbol"] = has_at
    if has_at:
        score += 0.20
        reasons.append("'@' in URL — browser ignores everything before @, classic phishing.")
    signal_row("@ Symbol",            has_at, flag=has_at)

    # Double slash in path
    dbl_slash = "//" in path
    signals["double_slash"] = dbl_slash
    if dbl_slash:
        score += 0.10
        reasons.append("Double '//' in path — may confuse redirect parsers.")
    signal_row("Double // in Path",   dbl_slash, flag=dbl_slash)

    # Hex encoding in host
    hex_enc = "%" in host
    signals["hex_in_host"] = hex_enc
    if hex_enc:
        score += 0.15
        reasons.append("Percent-encoded characters in host — obfuscation technique.")
    signal_row("Hex Encoding in Host", hex_enc, flag=hex_enc)

    # Punycode
    is_punycode = host.startswith("xn--") or any(p.startswith("xn--") for p in parts)
    signals["is_punycode"] = is_punycode
    if is_punycode:
        score += 0.20
        reasons.append("Punycode / IDN domain — possible Unicode lookalike attack.")
    signal_row("Punycode / IDN",      is_punycode, flag=is_punycode)

    # Non-standard port
    non_std_port = port not in (None, 80, 443)
    signals["non_standard_port"] = port
    if non_std_port:
        score += 0.10
        reasons.append(f"Non-standard port ({port}) — legitimate sites rarely use custom ports.")
    signal_row("Port",                port if port else "Default", flag=non_std_port)

    # Dashes
    dash_count = host.count("-")
    signals["dash_count"] = dash_count
    if dash_count >= 2:
        inc = min(0.15, dash_count * 0.05)
        score += inc
        reasons.append(f"Many hyphens ({dash_count}) — used to impersonate brands (e.g. secure-paypal-login.com).")
    signal_row("Hyphens in Domain",   dash_count, flag=dash_count >= 2, warn=dash_count == 1)

    # Digits in domain
    digits = sum(c.isdigit() for c in host.replace(".", ""))
    signals["digits_in_domain"] = digits
    if digits >= 3:
        score += 0.08
        reasons.append(f"Multiple digits ({digits}) in domain — possible typosquatting.")
    signal_row("Digits in Domain",    digits, warn=digits >= 3)

    # 4. Keywords
    section("4. Keyword Analysis")
    found_host_kw = [kw for kw in HOST_KEYWORDS if kw in host]
    signals["host_keywords"] = found_host_kw
    if found_host_kw:
        inc = min(0.25, len(found_host_kw) * 0.08)
        score += inc
        reasons.append(f"Suspicious keywords in host: {', '.join(found_host_kw)}.")
    signal_row("Keywords in Host",    found_host_kw if found_host_kw else "None", flag=bool(found_host_kw))

    path_query_str = (path + "?" + query).lower()
    found_path_kw  = [kw for kw in PATH_KEYWORDS if kw in path_query_str]
    signals["path_keywords"] = found_path_kw
    if found_path_kw:
        inc = min(0.15, len(found_path_kw) * 0.05)
        score += inc
        reasons.append(f"Suspicious keywords in path/query: {', '.join(found_path_kw)}.")
    signal_row("Keywords in Path",    found_path_kw if found_path_kw else "None", flag=bool(found_path_kw))

    # 5. Query params
    section("5. Query Parameters")
    params = parse_qs(query)
    param_count = len(params)
    signals["query_param_count"] = param_count
    if param_count >= 5:
        score += 0.07
        reasons.append(f"Many query params ({param_count}) — may be harvesting user data.")
    signal_row("Param Count",         param_count, warn=param_count >= 3, flag=param_count >= 5)
    for k, v in list(params.items())[:8]:
        signal_row(f"  ?{k}",         v[0][:60] if v else "", warn=True)

    if fragment:
        signal_row("Fragment (#...)",  fragment[:60], warn=True)

    # 6. DNS Resolution (optional — requires network)
    section("6. DNS Resolution  (live lookup)")
    try:
        ip_resolved = socket.gethostbyname(host)
        signals["resolved_ip"] = ip_resolved
        signal_row("Resolved IP",      ip_resolved)

        # Try reverse DNS
        try:
            rdns = socket.gethostbyaddr(ip_resolved)[0]
            signals["reverse_dns"] = rdns
            signal_row("Reverse DNS",  rdns)
        except Exception:
            signal_row("Reverse DNS",  "N/A", warn=True)

    except socket.gaierror:
        signals["resolved_ip"] = None
        score += 0.20
        reasons.append("Domain does not resolve — likely a brand-new or fake domain.")
        signal_row("DNS Resolution",   "FAILED — Domain may not exist!", flag=True)

    # 7. WHOIS / domain age (optional — needs python-whois)
    section("7. WHOIS / Domain Age  (optional)")
    try:
        import whois  # type: ignore
        w = whois.whois(domain_name)
        created = w.creation_date
        if isinstance(created, list):
            created = created[0]
        if created:
            age_days = (datetime.now() - created).days
            signals["domain_age_days"] = age_days
            is_new = age_days < 30
            if is_new:
                score += 0.25
                reasons.append(f"Domain is only {age_days} days old — very suspicious.")
            signal_row("Creation Date",  created.strftime("%Y-%m-%d"), flag=is_new)
            signal_row("Domain Age",     f"{age_days} days", flag=age_days < 30, warn=age_days < 180)
        else:
            signal_row("Creation Date",  "Not available", warn=True)

        registrar = getattr(w, "registrar", None) or "Unknown"
        signal_row("Registrar",          registrar)

    except ImportError:
        signal_row("python-whois",       "Not installed  →  pip install python-whois", warn=True)
    except Exception as e:
        signal_row("WHOIS lookup",       f"Failed: {e}", warn=True)

    # ═════════════════════════════════════════════════════════════════════════
    # FINAL SCORE
    # ═════════════════════════════════════════════════════════════════════════
    score = round(min(1.0, max(0.0, score)), 4)

    if score < 0.25:
        label = GREEN + BOLD  + "🟢  LOW RISK    — Likely Safe"       + RESET
    elif score < 0.55:
        label = YELLOW + BOLD + "🟡  MEDIUM RISK — Suspicious"         + RESET
    else:
        label = RED + BOLD    + "🔴  HIGH RISK   — Likely Phishing!"   + RESET

    print()
    print(BOLD + "═" * 60 + RESET)
    print(BOLD + "  FINAL SCORE" + RESET)
    separator()

    bar_filled = int(score * 40)
    bar = "█" * bar_filled + "░" * (40 - bar_filled)
    bar_colour = RED if score >= 0.55 else (YELLOW if score >= 0.25 else GREEN)
    print(f"  Score : {bar_colour}{score:.2f} / 1.00{RESET}")
    print(f"  Bar   : {bar_colour}[{bar}]{RESET}")
    print(f"  Label : {label}")

    if reasons:
        print()
        print(BOLD + "  Reasons:" + RESET)
        for i, r in enumerate(reasons, 1):
            flag_c = RED if score >= 0.55 else YELLOW
            print(f"    {flag_c}{i:2}. {r}{RESET}")

    print(BOLD + "═" * 60 + RESET)
    print(f"  {DIM}Method: URL heuristics only  |  No ML / no page visit{RESET}")
    print(BOLD + "═" * 60 + RESET)
    print()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(BOLD + CYAN + "\n  PhishFree — URL Heuristic Analyzer" + RESET)
    print(DIM   + "  Paste a URL and press Enter. Type 'exit' to quit.\n" + RESET)

    while True:
        try:
            raw = input(BOLD + "  URL > " + RESET).strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Bye!")
            sys.exit(0)

        if not raw:
            continue
        if raw.lower() in ("exit", "quit", "q"):
            print("  Bye!")
            sys.exit(0)

        analyze(raw)
