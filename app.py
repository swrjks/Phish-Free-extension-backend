# backend/app.py - PhishFree backend (complete with anchor fixes)
import os
import csv
import re
import json
import time
import unicodedata
from datetime import datetime
from collections import defaultdict, Counter
from urllib.parse import urlparse
import pytz
from dotenv import load_dotenv
from web3 import Web3
import hashlib
import logging
from hexbytes import HexBytes
import base64
import io
import numpy as np

# Import model classes (keep these imports as they are)
from cnn_model import CNNModel as CNNScorer
from gnn_model import GraphEngine
from ensemble import combine_scores
from llm_model import TextScorer

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from functools import lru_cache
from typing import Dict, Any, List, Tuple, Optional
import math




# Load environment variables from backend/.env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# Optional: import your model components
try:
    from llm_model import TextScorer
except Exception:
    TextScorer = None

try:
    from redirect import follow_redirects, get_cert_fingerprint
except Exception:
    def follow_redirects(url): return {"final_url": url, "hops": [], "status_code": None}
    def get_cert_fingerprint(url): return {"cert_fp": None}

try:
    from domain_info import domain_whois_info, domain_asn_info
except Exception:
    def domain_whois_info(hostname): return {}
    def domain_asn_info(hostname): return {}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AGG_PATH = os.path.join(BASE_DIR, "aggregate_log.csv")
AGG_LOG = os.path.join(os.path.dirname(__file__), "aggregate_log.csv")
FP_LOG = os.path.join(os.path.dirname(__file__), "false_positives.csv")
ANCHORS_PATH = os.path.join(os.path.dirname(__file__), "anchors.csv")
IST = pytz.timezone("Asia/Kolkata")

# NOTE: Model initializations moved below to after Flask `app` creation
# (they previously ran before `app = Flask(...)` which caused NameError on app.logger)

# --- Web3 setup ---
WEB3_RPC_URL = os.getenv("WEB3_RPC_URL") or ""
WEB3_PRIVATE_KEY = os.getenv("WEB3_PRIVATE_KEY") or os.getenv("PRIVATE_KEY") or ""

# normalize private key to include 0x prefix (helps avoid subtle signing errors)
if WEB3_PRIVATE_KEY and not WEB3_PRIVATE_KEY.startswith("0x"):
    WEB3_PRIVATE_KEY = "0x" + WEB3_PRIVATE_KEY

if not WEB3_PRIVATE_KEY:
    print("Warning: WEB3_PRIVATE_KEY not set. Anchors requiring a tx will fall back to test mode.")
elif len(WEB3_PRIVATE_KEY) < 66:
    print("Warning: WEB3_PRIVATE_KEY looks short; ensure it's the full 32-byte hex key (with or without 0x).")

# chain id default to Sepolia if not provided
try:
    WEB3_CHAIN_ID = int(os.getenv("WEB3_CHAIN_ID", "11155111"))
except Exception:
    WEB3_CHAIN_ID = 11155111

w3 = None
ACCOUNT = None
if WEB3_RPC_URL:
    try:
        w3 = Web3(Web3.HTTPProvider(WEB3_RPC_URL))
        print(f"✅ Connected to RPC: {WEB3_RPC_URL}")
    except Exception as e:
        print("Warning: failed to initialize Web3 provider:", e)
        w3 = None

if w3 and WEB3_PRIVATE_KEY:
    try:
        ACCOUNT = w3.eth.account.from_key(WEB3_PRIVATE_KEY)
        print("Account Address:", ACCOUNT.address)
    except Exception as e:
        print("Warning: invalid WEB3_PRIVATE_KEY:", e)
        ACCOUNT = None
else:
    if w3 and not WEB3_PRIVATE_KEY:
        print("Warning: Web3 provider available but private key missing; anchors will be recorded as test mode.")
    ACCOUNT = None

# -------------------------
# Small helpers: URL regex, client_ip, simple rate limiter
# -------------------------
# Extract http/https/www style URLs from free text
URL_REGEX = re.compile(r"(https?://[^\s,;\"']+|www\.[^\s,;\"']+)", re.IGNORECASE)

# Basic in-memory rate limiter (demo-only; not persistent)
# Structure: { ip_str: [timestamp1, timestamp2, ...] }
_RATE_LIMIT_STORE = {}
_RATE_LIMIT_LOCK = None  # placeholder if you want a threading.Lock
# config: allow up to `RATE_LIMIT_MAX` requests per `RATE_LIMIT_WINDOW` seconds
RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_MAX", "60"))       # default 60 reqs
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60")) # window in seconds

def client_ip():
    """
    Determine client IP from request, even behind a reverse proxy if X-Forwarded-For present.
    """
    try:
        forwarded = request.headers.get("X-Forwarded-For", "")
        if forwarded:
            # X-Forwarded-For may contain comma-separated list; take first
            return forwarded.split(",")[0].strip()
        if request.remote_addr:
            return request.remote_addr
        return request.environ.get("REMOTE_ADDR", "unknown")
    except Exception:
        return "unknown"

def is_rate_limited(ip: str) -> bool:
    """
    Very small in-memory sliding-window rate limiter.
    Returns True if the IP exceeded RATE_LIMIT_MAX requests in the last RATE_LIMIT_WINDOW seconds.
    Demo-only; use Redis/DB for production.
    """
    try:
        now = time.time()
        window_start = now - RATE_LIMIT_WINDOW
        hits = _RATE_LIMIT_STORE.get(ip, [])
        # keep only hits inside the window
        hits = [t for t in hits if t >= window_start]
        if len(hits) >= RATE_LIMIT_MAX:
            # update store to trimmed list to avoid unbounded growth
            _RATE_LIMIT_STORE[ip] = hits
            return True
        # record the new hit
        hits.append(now)
        _RATE_LIMIT_STORE[ip] = hits
        return False
    except Exception:
        # on any error, don't rate limit (fail-open for demo)
        return False


def extract_edges_from_aggregate(csv_path: str):
    """
    Parse aggregate_log.csv and produce a list of (src, dst) edges.
    Creates edges based on domain relationships, IP addresses, and URL patterns.
    """
    edges = []
    try:
        with open(csv_path, newline='', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            
            # Track domains, IPs, and their relationships
            domains = set()
            ips = set()
            domain_connections = {}
            ip_connections = {}
            domain_ip_map = {}
            
            for row in reader:
                # Extract domain from URL
                url = row.get("url", "").strip()
                domain = row.get("domain", "").strip()
                
                if not domain and url:
                    try:
                        parsed = urlparse(url)
                        domain = parsed.hostname or ""
                    except Exception:
                        continue
                
                if domain:
                    domains.add(domain)
                    
                    # Try to resolve IP for this domain
                    try:
                        import socket
                        ip = socket.gethostbyname(domain)
                        ips.add(ip)
                        domain_ip_map[domain] = ip
                        
                        # Group domains by IP
                        if ip not in ip_connections:
                            ip_connections[ip] = set()
                        ip_connections[ip].add(domain)
                    except Exception:
                        pass
                    
                    # Create connections based on similar domains or suspicious patterns
                    # Group similar domains together
                    base_domain = domain
                    if '.' in domain:
                        parts = domain.split('.')
                        if len(parts) >= 2:
                            base_domain = '.'.join(parts[-2:])  # Get TLD + domain
                    
                    if base_domain not in domain_connections:
                        domain_connections[base_domain] = set()
                    domain_connections[base_domain].add(domain)
            
            # Create edges between domains sharing the same IP
            for ip, domain_set in ip_connections.items():
                domain_list = list(domain_set)
                if len(domain_list) > 1:
                    # Connect all domains sharing the same IP
                    for i in range(len(domain_list)):
                        for j in range(i + 1, len(domain_list)):
                            edges.append((domain_list[i], domain_list[j]))
                            edges.append((domain_list[j], domain_list[i]))  # Bidirectional
            
            # Create edges between domains in the same group (same base domain)
            for base_domain, domain_set in domain_connections.items():
                domain_list = list(domain_set)
                if len(domain_list) > 1:
                    # Connect all domains in the same group
                    for i in range(len(domain_list)):
                        for j in range(i + 1, len(domain_list)):
                            edges.append((domain_list[i], domain_list[j]))
                            edges.append((domain_list[j], domain_list[i]))  # Bidirectional
            
            # Add comprehensive synthetic suspicious domain connections for better GNN performance
            suspicious_patterns = [
                # Banking/Financial phishing
                ('secure-bank-login.example', 'fake-bank.com'),
                ('bank-verify.net', 'account-update.org'),
                ('payment-secure.com', 'billing-service.net'),
                ('login-verify.org', 'account-security.com'),
                
                # Crypto/Investment scams
                ('crypto-wallet.org', 'bitcoin-exchange.com'),
                ('investment-opportunity.net', 'crypto-mining.org'),
                ('wallet-recovery.com', 'blockchain-verify.net'),
                
                # Social media/Account verification
                ('social-login.net', 'profile-update.com'),
                ('account-verify.org', 'security-check.net'),
                ('login-secure.com', 'verify-account.org'),
                
                # Payment/Invoice scams
                ('payment-update.com', 'invoice-pay.net'),
                ('billing-service.org', 'payment-verify.com'),
                ('invoice-secure.net', 'payment-gateway.org'),
                
                # Tech support scams
                ('tech-support.net', 'system-update.org'),
                ('security-alert.com', 'virus-removal.net'),
                ('windows-update.org', 'microsoft-support.com'),
                
                # Email/Communication
                ('email-verify.net', 'account-suspended.org'),
                ('mail-security.com', 'inbox-update.net'),
                ('message-center.org', 'notification-service.com')
            ]
            
            # Add synthetic connections for domains that exist in our data
            for src, dst in suspicious_patterns:
                if src in domains or dst in domains:
                    edges.append((src, dst))
                    edges.append((dst, src))
            
            # Add connections based on suspicious keywords in domain names
            suspicious_keywords = ['secure', 'login', 'verify', 'account', 'bank', 'payment', 'crypto', 'wallet']
            for domain in domains:
                for keyword in suspicious_keywords:
                    if keyword in domain.lower():
                        # Connect to other domains with similar keywords
                        for other_domain in domains:
                            if other_domain != domain and keyword in other_domain.lower():
                                edges.append((domain, other_domain))
                                edges.append((other_domain, domain))
            
            # De-duplicate edges
            edges = list({(a, b) for (a, b) in edges if a and b and a != b})
            
    except Exception as e:
        # app.logger not available here yet; print as fallback
        try:
            print("Failed to extract edges from aggregate CSV:", e)
        except Exception:
            pass
    return edges


# --- Anchors CSV helpers ---
def ensure_anchors_csv():
    header = ["timestamp", "mode", "rows", "first_row_index", "last_row_index",
              "batch_hash", "tx_hash", "chain_id", "status", "message"]
    if not os.path.exists(ANCHORS_PATH):
        with open(ANCHORS_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

def append_anchor_row(row: dict):
    ensure_anchors_csv()
    header = ["timestamp", "mode", "rows", "first_row_index", "last_row_index",
              "batch_hash", "tx_hash", "chain_id", "status", "message"]
    with open(ANCHORS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writerow({k: row.get(k, "") for k in header})

# --- Flask setup ---
app = Flask(__name__)
CORS(app)

# configure logging for easier debugging
logging.basicConfig(level=logging.DEBUG)
app.logger.setLevel(logging.DEBUG)


# initialize scorer if present
# NOTE: create both `scorer` and `text_scorer` so existing endpoints that use either name keep working
# -----------------------------
text_scorer = None
scorer = None
try:
    if TextScorer:
        text_scorer = TextScorer()
        scorer = text_scorer
        app.logger.info("TextScorer initialized")
    else:
        scorer = None
        text_scorer = None
        app.logger.info("TextScorer not available; using heuristics")
except Exception as e:
    app.logger.warning("TextScorer init failed: %s", e)
    text_scorer = None
    scorer = None

# initialize CNN scorer
cnn_scorer = None
try:
    cnn_scorer = CNNScorer()
    templates_dir = os.path.join(os.path.dirname(__file__), "templates")
    if os.path.exists(templates_dir):
        cnn_scorer.compute_brand_embeddings(templates_dir)
    app.logger.info("CNNScorer initialized successfully")
except Exception as e:
    app.logger.warning("CNNScorer init failed: %s", e)
    cnn_scorer = None

# initialize GraphEngine
graph_engine = None
try:
    graph_engine = GraphEngine()
    
    # Initialize with some test data to make GNN work
    test_edges = [
        ('test-phish.com', 'phishing-site.com'),
        ('fake-bank.com', 'phishing-site.com'),
        ('suspicious-domain.net', 'malicious-page.org')
    ]
    graph_engine.build_graph_from_edges(test_edges)
    graph_engine.compute_node2vec_embeddings(dimensions=32, walk_length=5, num_walks=20)
    app.logger.info("GraphEngine initialized successfully")
except Exception as e:
    app.logger.warning("GraphEngine init failed: %s", e)
    graph_engine = None

# --- Simple caching for performance ---
text_cache = {}
cnn_cache = {}
gnn_cache = {}

# --- Additional startup info for debugging ---
app.logger.info("=== Model availability on startup ===")
app.logger.info("TextScorer available: %s", bool(text_scorer))
app.logger.info("CNNScorer available: %s", bool(cnn_scorer))
app.logger.info("GraphEngine available: %s", bool(graph_engine))

# --- Build graph once at startup (use wrapper API) ---
if graph_engine is not None:
    try:
        AGG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aggregate_log.csv")
        edges = extract_edges_from_aggregate(AGG_PATH)
        if edges:
            app.logger.info("Building graph from aggregate (edges=%d)", len(edges))
            graph_engine.build_graph_from_edges(edges)
            graph_engine.compute_node2vec_embeddings(dimensions=64, walk_length=10, num_walks=80)
            try:
                n_nodes = len(getattr(graph_engine, "embeddings", {}) or {})
            except Exception:
                n_nodes = 0
            app.logger.info("Graph embeddings computed (nodes=%d)", n_nodes)
        else:
            app.logger.info("No edges found in aggregate_log.csv; graph left empty")
    except Exception as e:
        app.logger.warning("GraphEngine build failed: %s", e)


# --- Dashboard route ---
@app.route("/dashboard", methods=["GET"])
def dashboard():
    """Serve the dashboard HTML file"""
    return send_file(os.path.join(os.path.dirname(__file__), "static", "dashboard.html"))

@app.route("/", methods=["GET"])
def root():
    """Redirect root to dashboard"""
    return send_file(os.path.join(os.path.dirname(__file__), "static", "dashboard.html"))

@app.route("/aggregate/has_log", methods=["GET"])
def has_log():
    """Check if aggregate log exists"""
    return jsonify({"exists": os.path.exists(AGG_LOG)})

@app.route("/report/false_positive", methods=["POST"])
def report_false_positive():
    """Report a site as safe (false positive)"""
    try:
        data = request.get_json(force=True)
        url = data.get("url", "")
        hostname = data.get("hostname", "")
        timestamp = data.get("timestamp", datetime.utcnow().isoformat() + "Z")
        note = data.get("note", "reported_safe_by_user")
        analysis = data.get("analysis", {})
        
        # Create false positives CSV if it doesn't exist
        false_positives_path = os.path.join(os.path.dirname(__file__), "false_positives.csv")
        
        # Check if file exists and has headers
        file_exists = os.path.exists(false_positives_path)
        if not file_exists:
            with open(false_positives_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "url", "hostname", "note", "analysis_json"])
        
        # Append the false positive report
        with open(false_positives_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, url, hostname, note, json.dumps(analysis)])
        
        app.logger.info("False positive reported for URL: %s", url)
        return jsonify({"ok": True, "message": "Site reported as safe"})
        
    except Exception as e:
        app.logger.exception("Failed to report false positive: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500

# --- Health + ping ---
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "OK", "service": "phish-free-backend"})

@app.route("/ping", methods=["POST"])
def ping():
    data = request.get_json(silent=True) or {}
    return jsonify({"pong": True, "received": data})

# --- Legacy Wrappers ---
@app.route("/analyze/aggregate", methods=["POST"])
def analyze_aggregate_legacy():
    return analyze_multi()

@app.route("/analyze/text", methods=["POST"])
def analyze_text_legacy():
    return analyze_multi()

@app.route("/analyze/url", methods=["POST"])
def analyze_url_legacy():
    return analyze_multi()

# --- Report + top endpoints ---
@app.route("/aggregate/report", methods=["GET"])
def aggregate_report():
    fmt=request.args.get("format","json").lower()
    if not os.path.exists(AGG_LOG): return jsonify({"error":"No log file yet"}),404
    if fmt=="csv":
        try: return send_file(AGG_LOG,mimetype="text/csv",as_attachment=True,download_name=os.path.basename(AGG_LOG))
        except: return send_file(AGG_LOG,mimetype="text/csv",as_attachment=True,attachment_filename=os.path.basename(AGG_LOG))
    else:
        data = []
        with open(AGG_LOG,"r",encoding="utf-8") as f:
            lines = f.readlines()
        
        # Process each line and detect format
        for i, line in enumerate(lines):
            if i == 0:  # Skip header
                continue
            if not line.strip():
                continue
                
            # Parse CSV line - handle quoted fields properly
            import csv as csv_module
            from io import StringIO
            reader = csv_module.reader(StringIO(line))
            try:
                row = next(reader)
            except:
                continue
            
            # Detect format based on number of columns and content
            if len(row) >= 9:  # New format: timestamp,url,domain,text_score,cnn_score,gnn_score,combined_score,label,text_excerpt,combined_reasons
                try:
                    normalized_row = {
                        "timestamp": row[0] if len(row) > 0 else "",
                        "url": row[1] if len(row) > 1 else "",
                        "domain": row[2] if len(row) > 2 else "",
                        "text_score": row[3] if len(row) > 3 else "0",
                        "cnn_score": row[4] if len(row) > 4 else "0",
                        "gnn_score": row[5] if len(row) > 5 else "0",
                        "combined_score": row[6] if len(row) > 6 else "0",
                        "label": row[7] if len(row) > 7 else "",
                        "text_excerpt": row[8] if len(row) > 8 else "",
                        "combined_reasons": row[9] if len(row) > 9 else "",
                        "screenshot_url": row[10] if len(row) > 10 else ""
                    }
                except Exception:
                    # Fallback for malformed lines
                    normalized_row = {
                        "timestamp": "", "url": "", "domain": "", "text_score": "0", 
                        "cnn_score": "0", "gnn_score": "0", "combined_score": "0", 
                        "label": "", "text_excerpt": "", "combined_reasons": ""
                    }
            else:  # Old format: timestamp,text_score,url_score,aggregate_score,label,badge,text_excerpt,url,combined_reasons
                try:
                    normalized_row = {
                        "timestamp": row[0] if len(row) > 0 else "",
                        "url": row[7] if len(row) > 7 else "",
                        "domain": "",
                        "text_score": row[1] if len(row) > 1 else "0",
                        "cnn_score": "0",  # Not available in old format - set to 0
                        "gnn_score": "0",  # Not available in old format - set to 0
                        "combined_score": row[3] if len(row) > 3 else "0",  # aggregate_score -> combined_score
                        "label": row[4] if len(row) > 4 else "",
                        "text_excerpt": row[6] if len(row) > 6 else "",
                        "combined_reasons": row[8] if len(row) > 8 else "",
                        "screenshot_url": ""
                    }
                except Exception:
                    # Fallback for malformed lines
                    normalized_row = {
                        "timestamp": "", "url": "", "domain": "", "text_score": "0", 
                        "cnn_score": "0", "gnn_score": "0", "combined_score": "0", 
                        "label": "", "text_excerpt": "", "combined_reasons": ""
                    }
            
            data.append(normalized_row)
        return jsonify(data)

@app.route("/aggregate/report", methods=["DELETE"])
def clear_logs():
    if not os.path.exists(AGG_LOG):
        return jsonify({"ok": True, "message": "No logs to clear"})
        
    ts_to_delete = request.args.get("timestamp")
    try:
        if ts_to_delete:
            with open(AGG_LOG, "r", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            
            # Keep header (index 0) and rows that do not match the target timestamp
            new_rows = [row for i, row in enumerate(rows) if i == 0 or (len(row) > 0 and row[0] != ts_to_delete)]
            
            with open(AGG_LOG, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(new_rows)
            return jsonify({"ok": True, "message": f"Log entry {ts_to_delete} deleted successfully"})
        else:
            headers = ["timestamp", "url", "domain", "text_score", "cnn_score", "gnn_score", "combined_score", "label", "text_excerpt", "combined_reasons", "screenshot_url"]
            with open(AGG_LOG, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(headers)
            return jsonify({"ok": True, "message": "Logs cleared successfully"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/aggregate/top_domains", methods=["GET"])
def top_domains():
    if not os.path.exists(AGG_LOG): return jsonify({"data":[]})
    domains=[]
    with open(AGG_LOG,"r",encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                h=urlparse(r.get("url") or "").hostname or ""
                if h: domains.append(h.lower())
            except: continue
    cnt=Counter(domains).most_common(30)
    return jsonify({"data":[{"domain":d,"count":c} for d,c in cnt]})

@app.route("/aggregate/top_keywords", methods=["GET"])
def top_keywords():
    if not os.path.exists(AGG_LOG): return jsonify({"data":[]})
    kws=[]; kw_re=re.compile(r"\b[a-z]{3,}\b",re.I)
    with open(AGG_LOG,"r",encoding="utf-8") as f:
        for r in csv.DictReader(f):
            words=kw_re.findall((r.get("combined_reasons") or "")+" "+str(r.get("text_score") or ""))
            kws+=[w.lower() for w in words]
    cnt=Counter(kws).most_common(40)
    return jsonify({"data":[{"keyword":k,"count":c} for k,c in cnt]})

# --- Anchors endpoints ---
@app.route("/aggregate/anchors", methods=["GET"])
def aggregate_anchors():
    """
    Return anchors.csv as JSON but sanitize any CSV rows which may contain
    an extra None-key (csv.DictReader places extra columns under None).
    """
    if not os.path.exists(ANCHORS_PATH):
        return jsonify({"data": []})

    sanitized = []
    try:
        with open(ANCHORS_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                # If DictReader put extra columns under None, move them into a safe key
                if None in row:
                    extra = row.pop(None)
                    if isinstance(extra, list):
                        extra_val = ", ".join([str(x) for x in extra if x is not None and str(x).strip() != ""])
                    else:
                        extra_val = str(extra)
                    row["_extra_columns"] = extra_val

                # Ensure all keys are strings (avoid any accidental None keys)
                clean_row = {}
                for k, v in (row.items()):
                    key = str(k) if k is not None else "_none_key_"
                    clean_row[key] = v
                sanitized.append(clean_row)
    except Exception as e:
        app.logger.exception("Failed to read/sanitize anchors.csv: %s", e)
        return jsonify({"error": f"Failed to read anchors.csv: {e}"}), 500


    return jsonify({"data": sanitized})


def compute_eip1559_fees(preferred_priority_gwei=2):
    """
    Compute (maxPriorityFeePerGas, maxFeePerGas) in wei, ensuring
    maxFeePerGas >= baseFee + maxPriorityFeePerGas when baseFee is available.
    Returns (maxPriority, maxFee, base_fee_or_none)
    """
    # priority in wei
    max_priority = int(preferred_priority_gwei * (10**9))
    base_fee = None
    try:
        # try pending block first, then latest
        try:
            blk = w3.eth.get_block('pending')
        except Exception:
            blk = w3.eth.get_block('latest')
        base_fee = blk.get('baseFeePerGas', None)
    except Exception:
        base_fee = None

    # fallback gas price (legacy) if base_fee not available
    try:
        legacy_gas_price = w3.eth.gas_price or w3.to_wei(20, 'gwei')
    except Exception:
        legacy_gas_price = w3.to_wei(20, 'gwei')

    if base_fee is not None and base_fee > 0:
        # ensure maxFee >= base_fee + priority + small padding
        padding = max(int(base_fee * 0.10), int(max_priority * 2))
        max_fee = base_fee + max_priority + padding
    else:
        # no base_fee info -> use legacy gas price heuristic but ensure max_fee > priority
        max_fee = max(int(legacy_gas_price * 2.5), max_priority + int(legacy_gas_price))
        if max_fee <= max_priority:
            max_fee = max_priority + int(legacy_gas_price)

    # final guard
    if max_fee <= max_priority:
        max_fee = max_priority + 1

    return max_priority, max_fee, base_fee


@app.route("/aggregate/anchor", methods=["POST"])
def create_anchor():
    """
    Robust anchor endpoint:
    POST JSON: { "n":50, "test_mode": false, "wait": true/false }
    Returns JSON with ok, batch_hash, tx_hash (if any), chain_id, first_row_index, last_row_index, error/message
    """
    data = request.get_json(force=True, silent=True) or {}
    try:
        n = int(data.get("n", 50))
    except Exception:
        n = 50
    test_mode = bool(data.get("test_mode", False))
    wait_for_receipt = bool(data.get("wait", False))

    ts = datetime.utcnow().isoformat() + "Z"

    # Ensure aggregate log exists
    if not os.path.exists(AGG_LOG):
        return jsonify({"ok": False, "error": "aggregate_log.csv missing"}), 400

    # Read AGG_LOG and skip header row if present
    with open(AGG_LOG, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]

    # If first line looks like a CSV header (contains comma-separated column names), skip it for hashing/indexing
    data_lines = lines[:]
    if data_lines and "," in data_lines[0] and any(h in data_lines[0].lower() for h in ("timestamp", "aggregate_score", "label")):
        data_lines = data_lines[1:]

    if not data_lines:
        return jsonify({"ok": False, "error": "No data rows in aggregate_log.csv"}), 400

    last_n = data_lines[-n:]
    # compute batch hash over the textual CSV lines (data rows only)
    batch_hash = hashlib.sha256("\n".join(last_n).encode("utf-8")).hexdigest()
    first_idx = max(1, len(data_lines) - len(last_n) + 1)
    last_idx = len(data_lines)

    # If test mode OR web3/account not ready -> record test anchor
    if test_mode or not (w3 and ACCOUNT):
        row = {
            "timestamp": ts,
            "mode": "test",
            "rows": n,
            "first_row_index": first_idx,
            "last_row_index": last_idx,
            "batch_hash": batch_hash,
            "tx_hash": "",
            "chain_id": "",
            "status": "test",
            "message": "test mode anchor (no blockchain tx)"
        }
        try:
            append_anchor_row(row)
        except Exception as e:
            return jsonify({"ok": False, "error": f"failed to write anchors.csv: {e}", **row}), 500
        return jsonify({**row, "ok": True}), 200

    # Build and send transaction (robust)
    try:
        # Use pending transaction count to avoid reusing nonce that are pending in the node
        base_nonce = w3.eth.get_transaction_count(ACCOUNT.address, "pending")

        # Build common unsigned tx template
        def build_unsigned(nonce_val):
            tx = {
                "to": ACCOUNT.address,
                "value": 0,
                "nonce": nonce_val,
                "chainId": WEB3_CHAIN_ID,
            }
            # include batch hash as hex data (use plain "0x..." string)
            tx["data"] = "0x" + batch_hash if batch_hash else "0x"
            return tx

        # We'll try to sign/send up to `max_attempts` nonces (in case pending txs cause collisions)
        max_attempts = 4
        last_exception = None
        tx_hash = None

        for attempt in range(max_attempts):
            nonce_to_try = base_nonce + attempt
            unsigned_tx = build_unsigned(nonce_to_try)

            # Estimate gas (safe fallback)
            try:
                estimated = w3.eth.estimate_gas(unsigned_tx)
                unsigned_tx["gas"] = max(21000, int(estimated * 1.2))
            except Exception as e_est:
                unsigned_tx["gas"] = 120000
                app.logger.debug("Gas estimate failed, using fallback 120000: %s", e_est)

            # Compute safe EIP-1559 fees
            try:
                max_priority, max_fee, base_fee = compute_eip1559_fees(preferred_priority_gwei=2)
                unsigned_tx["maxPriorityFeePerGas"] = max_priority
                unsigned_tx["maxFeePerGas"] = max_fee
                unsigned_tx.pop("gasPrice", None)
            except Exception as e_fee:
                # fallback to legacy gasPrice but ensure gasPrice >= 1 wei over priority
                try:
                    gp = w3.eth.gas_price or w3.to_wei(20, 'gwei')
                except Exception:
                    gp = w3.to_wei(20, 'gwei')
                unsigned_tx["gasPrice"] = gp

            app.logger.debug("Attempting tx with nonce %s: %s", nonce_to_try, unsigned_tx)

            # === robust sign & send block ===
            try:
                signed = w3.eth.account.sign_transaction(unsigned_tx, private_key=WEB3_PRIVATE_KEY)

                # robustly extract raw transaction payload (support multiple web3/eth-account shapes)
                raw_tx = None
                if hasattr(signed, "rawTransaction"):
                    raw_tx = getattr(signed, "rawTransaction")
                elif hasattr(signed, "raw_transaction"):
                    raw_tx = getattr(signed, "raw_transaction")
                elif isinstance(signed, dict):
                    raw_tx = signed.get("rawTransaction") or signed.get("raw_transaction") or signed.get("raw_tx") or signed.get("rawTx")
                elif isinstance(signed, (bytes, bytearray, str)):
                    raw_tx = signed

                if raw_tx is None:
                    app.logger.error("Signed transaction object missing raw payload: %r", signed)
                    raise Exception("signed transaction object missing raw payload (no rawTransaction/raw_transaction)")

                # normalize to HexBytes for send_raw_transaction
                if isinstance(raw_tx, str):
                    raw_hex = raw_tx if raw_tx.startswith("0x") else "0x" + raw_tx
                    raw_bytes = HexBytes(raw_hex)
                else:
                    raw_bytes = HexBytes(raw_tx)

                sent = w3.eth.send_raw_transaction(raw_bytes)
                tx_hash = sent.hex() if hasattr(sent, "hex") else str(sent)

                row = {
                    "timestamp": ts,
                    "mode": "real",
                    "rows": n,
                    "first_row_index": first_idx,
                    "last_row_index": last_idx,
                    "batch_hash": batch_hash,
                    "tx_hash": tx_hash,
                    "chain_id": WEB3_CHAIN_ID,
                    "status": "submitted",
                    "message": f"tx submitted (nonce={nonce_to_try})"
                }
                try:
                    append_anchor_row(row)
                except Exception as e_app:
                    app.logger.error("append_anchor_row failed after tx send: %s", e_app)
                    return jsonify({"ok": True, "tx_hash": tx_hash, "warning": f"tx_sent_but_append_failed: {e_app}", **row}), 200

                # optionally wait for receipt
                if wait_for_receipt:
                    try:
                        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
                        row["status"] = "confirmed" if receipt and receipt.status == 1 else "failed"
                        row["message"] = f"receipt status={getattr(receipt,'status',None)} block={getattr(receipt,'blockNumber',None)}"
                        append_anchor_row(row)
                    except Exception as e_wait:
                        app.logger.warning("wait_for_receipt failed: %s", e_wait)
                        return jsonify({"ok": True, "tx_hash": tx_hash, "message": "tx submitted; receipt wait failed", "error_wait": str(e_wait), **row}), 200

                # success - return
                return jsonify({"ok": True, "batch_hash": batch_hash, "tx_hash": tx_hash, "chain_id": WEB3_CHAIN_ID,
                                "first_row_index": first_idx, "last_row_index": last_idx}), 200

            except Exception as e_send:
                last_exception = e_send
                err_s = str(e_send).lower()
                app.logger.warning("send_raw_transaction attempt nonce=%s failed: %s", nonce_to_try, e_send)

                # If node says tx already known, it might be the same signed tx or same nonce already used.
                if "already known" in err_s or "known transaction" in err_s:
                    # Try next nonce (there may be a pending tx). continue loop to retry with next nonce.
                    continue

                # If node rejects due to priority/fee mismatch, try re-computing fees and retry (same nonce)
                if "max priority fee per gas higher than max fee per gas" in err_s or "replacement transaction underpriced" in err_s or "underpriced" in err_s:
                    app.logger.info("Fee-related error detected, attempting fee-bump and retry on same nonce.")
                    try:
                        # recompute fees with a higher priority
                        bump_priority_gwei = 6  # bump priority for retry
                        max_priority_b, max_fee_b, base_fee_b = compute_eip1559_fees(preferred_priority_gwei=bump_priority_gwei)
                        unsigned_tx["maxPriorityFeePerGas"] = max_priority_b
                        unsigned_tx["maxFeePerGas"] = max_fee_b
                        unsigned_tx.pop("gasPrice", None)

                        signed = w3.eth.account.sign_transaction(unsigned_tx, private_key=WEB3_PRIVATE_KEY)

                        raw_tx = None
                        if hasattr(signed, "rawTransaction"):
                            raw_tx = getattr(signed, "rawTransaction")
                        elif hasattr(signed, "raw_transaction"):
                            raw_tx = getattr(signed, "raw_transaction")
                        elif isinstance(signed, dict):
                            raw_tx = signed.get("rawTransaction") or signed.get("raw_transaction") or signed.get("raw_tx") or signed.get("rawTx")
                        elif isinstance(signed, (bytes, bytearray, str)):
                            raw_tx = signed

                        if raw_tx is None:
                            raise Exception("signed transaction object missing raw payload on fee-bump")

                        if isinstance(raw_tx, str):
                            raw_hex = raw_tx if raw_tx.startswith("0x") else "0x" + raw_tx
                            raw_bytes = HexBytes(raw_hex)
                        else:
                            raw_bytes = HexBytes(raw_tx)

                        sent = w3.eth.send_raw_transaction(raw_bytes)
                        tx_hash = sent.hex() if hasattr(sent, "hex") else str(sent)
                        row = {
                            "timestamp": ts,
                            "mode": "real",
                            "rows": n,
                            "first_row_index": first_idx,
                            "last_row_index": last_idx,
                            "batch_hash": batch_hash,
                            "tx_hash": tx_hash,
                            "chain_id": WEB3_CHAIN_ID,
                            "status": "submitted",
                            "message": f"tx submitted after fee bump (nonce={nonce_to_try})"
                        }
                        append_anchor_row(row)
                        return jsonify({"ok": True, "batch_hash": batch_hash, "tx_hash": tx_hash, "chain_id": WEB3_CHAIN_ID,
                                        "first_row_index": first_idx, "last_row_index": last_idx}), 200
                    except Exception as e2:
                        app.logger.warning("retry with bumped fees failed: %s", e2)
                        last_exception = e2
                        continue

                # Otherwise break and return the error
                break

        # If we exhausted attempts
        err_msg = str(last_exception) if last_exception else "unknown error sending tx"
        app.logger.error("Failed to send tx after %s attempts: %s", max_attempts, err_msg)
        # append failed row for audit
        try:
            append_anchor_row({
                "timestamp": ts,
                "mode": "real",
                "rows": n,
                "first_row_index": first_idx,
                "last_row_index": last_idx,
                "batch_hash": batch_hash,
                "tx_hash": "",
                "chain_id": WEB3_CHAIN_ID,
                "status": "failed",
                "message": err_msg[:1000]
            })
        except Exception:
            app.logger.exception("Failed to append failed anchor row")
        # If we saw an "already known" style message, return 400 with that explanation
        if last_exception and ("already known" in str(last_exception).lower() or "known transaction" in str(last_exception).lower()):
            return jsonify({"ok": False, "error": "tx already known (duplicate/pending)", "batch_hash": batch_hash,
                            "first_row_index": first_idx, "last_row_index": last_idx}), 400

        return jsonify({"ok": False, "error": err_msg, "batch_hash": batch_hash,
                        "first_row_index": first_idx, "last_row_index": last_idx}), 500

    except Exception as e:
        err_msg = str(e)
        app.logger.exception("Unexpected error sending anchor tx: %s", err_msg)
        try:
            append_anchor_row({
                "timestamp": ts,
                "mode": "real",
                "rows": n,
                "first_row_index": first_idx,
                "last_row_index": last_idx,
                "batch_hash": batch_hash,
                "tx_hash": "",
                "chain_id": WEB3_CHAIN_ID,
                "status": "failed",
                "message": err_msg[:1000]
            })
        except Exception:
            app.logger.exception("Failed to append failed anchor row")
        return jsonify({"ok": False, "error": err_msg, "batch_hash": batch_hash,
                        "first_row_index": first_idx, "last_row_index": last_idx}), 500
    
@app.route("/analyze/multi", methods=["POST"])
def analyze_multi():
    """
    Body JSON:
      { "text": "...", "image_b64": "<base64>", "domain": "example.com" }
    Returns JSON with combined score + components + reasons.
    """
    # Trusted Major Apps (AI Agent Whitelist)
    MAJOR_SITES = {
        "google.com", "www.google.com", "youtube.com", "www.youtube.com",
        "spotify.com", "open.spotify.com", "facebook.com", "www.facebook.com",
        "instagram.com", "www.instagram.com", "linkedin.com", "www.linkedin.com",
        "github.com", "www.github.com", "microsoft.com", "www.microsoft.com",
        "apple.com", "www.apple.com", "amazon.com", "www.amazon.com",
        "netflix.com", "www.netflix.com", "twitter.com", "x.com", "www.x.com"
    }
    
    # Official brand domains for impersonation detection
    OFFICIAL_DOMAINS = {
        "google": ["google.com", "google.co.in", "youtube.com"],
        "microsoft": ["microsoft.com", "office.com"],
        "amazon": ["amazon.com", "amazon.in", "amazon.co.uk"],
        "paypal": ["paypal.com"],
        "hsbc": ["hsbc.com", "hsbc.co.uk"],
        "chase": ["chase.com"],
        "wellsfargo": ["wellsfargo.com"],
        "hdfc": ["hdfcbank.com"],
        "sbi": ["onlinesbi.sbi", "sbi.co.in"]
    }

    try:
        data = request.get_json(force=True)
    except Exception as e:
        return jsonify({"ok": False, "error": "invalid json: " + str(e)}), 400

    text = data.get("text", "") or ""
    domain = data.get("domain", "") or ""
    image_b64 = data.get("image_b64", None)

    # Initialize all analysis variables to prevent UnboundLocalError
    tscore = 0.0
    t_res = {"score": 0.0, "reasons": []}
    cscore = None
    c_reasons = []
    raw_c_res = None
    gscore = None
    g_reasons = []
    agent_reasons = []
    agent_override_score = None
    screenshot_url = None

    # 1. Text score (llm_model.TextScorer) with caching
    if text:
        text_key = f"{text[:100]}"
        if text_key in text_cache:
            t_res = text_cache[text_key]
            tscore = float(t_res.get("score", 0.0))
        else:
            try:
                if text_scorer:
                    t_res = text_scorer.score(text)
                    tscore = float(t_res.get("score", 0.0))
                    text_cache[text_key] = t_res
                    if len(text_cache) > 200: text_cache.clear()
            except Exception as e:
                app.logger.error("Text scoring failed: %s", e)
                t_res = {"score": 0.0, "reasons": [str(e)]}

    # 2. CNN score
    if image_b64:
        if cnn_scorer is None:
            c_reasons = ["cnn model not loaded"]
            cscore = None
            app.logger.debug("analyze/multi: image provided but cnn_scorer is not initialized")
        else:
            try:
                # Accept either full data-uri or plain base64
                img_b64 = image_b64
                if "," in img_b64:
                    img_b64 = img_b64.split(",", 1)[1]
                image_bytes = base64.b64decode(img_b64)
                
                # Optimize image size for faster processing
                try:
                    from PIL import Image
                    import io
                    img = Image.open(io.BytesIO(image_bytes))
                    # Resize large images to max 512x512 for faster processing
                    if img.size[0] > 512 or img.size[1] > 512:
                        img.thumbnail((512, 512), Image.Resampling.LANCZOS)
                        img_bytes = io.BytesIO()
                        img.save(img_bytes, format='PNG', optimize=True)
                        image_bytes = img_bytes.getvalue()
                        app.logger.debug("analyze/multi: optimized image size to %dx%d", img.size[0], img.size[1])
                except Exception as e:
                    app.logger.debug("analyze/multi: image optimization failed: %s", e)

                # --- Save Screenshot to static/screenshots ---
                try:
                    safe_domain = re.sub(r'[^\w\.-]', '_', domain) if domain else "unknown"
                    fname = f"ss_{int(time.time())}_{safe_domain}.png"
                    static_dir = os.path.join(os.path.dirname(__file__), "static", "screenshots")
                    if not os.path.exists(static_dir):
                        os.makedirs(static_dir, exist_ok=True)
                    fpath = os.path.join(static_dir, fname)
                    with open(fpath, "wb") as f:
                        f.write(image_bytes)
                    screenshot_url = f"/static/screenshots/{fname}"
                    app.logger.debug("analyze/multi: saved screenshot to %s", screenshot_url)
                except Exception as e:
                    app.logger.error("analyze/multi: image save failed: %s", e)
                
                app.logger.debug("analyze/multi: decoded image bytes len=%d", len(image_bytes))

                # call wrapped scorer and log the raw response for diagnosis
                raw_c_res = cnn_scorer.score_image_bytes(image_bytes)
                app.logger.debug("analyze/multi: raw cnn response: %r", raw_c_res)

                # Normalize different possible outputs:
                parsed_score = None
                if isinstance(raw_c_res, (int, float)):
                    parsed_score = float(raw_c_res)
                    c_reasons = []
                elif isinstance(raw_c_res, dict):
                    # prefer explicit 'score' key
                    if "score" in raw_c_res and raw_c_res.get("score") is not None:
                        try:
                            parsed_score = float(raw_c_res.get("score"))
                        except Exception:
                            parsed_score = None
                    # fallback: scan for a plausible numeric field
                    if parsed_score is None:
                        for k, v in raw_c_res.items():
                            if k in ("ok", "reasons", "error"):
                                continue
                            if isinstance(v, (int, float)):
                                parsed_score = float(v); break
                            if isinstance(v, str):
                                try:
                                    fv = float(v); parsed_score = fv; break
                                except Exception:
                                    pass
                    # reasons extraction
                    maybe_reasons = raw_c_res.get("reasons", None)
                    if isinstance(maybe_reasons, list):
                        c_reasons = maybe_reasons
                    elif isinstance(maybe_reasons, str):
                        c_reasons = [maybe_reasons]
                    else:
                        c_reasons = c_reasons or []
                else:
                    app.logger.warning("analyze/multi: cnn returned unknown shape: %r", type(raw_c_res))
                    parsed_score = None
                    c_reasons = ["unknown cnn response shape"]

                # clamp/normalize parsed_score to 0..1 if it looks like a percentage or >1
                if parsed_score is not None:
                    if parsed_score > 1.0:
                        if 0.0 < parsed_score <= 1000.0:
                            if parsed_score <= 100:
                                parsed_score = parsed_score / 100.0
                            elif parsed_score <= 1000:
                                parsed_score = parsed_score / 100.0
                            else:
                                parsed_score = parsed_score / parsed_score
                    parsed_score = max(0.0, min(1.0, float(parsed_score)))
                    cscore = parsed_score
                else:
                    cscore = None
                    if not c_reasons:
                        c_reasons = ["cnn returned no numeric score"]
            except Exception as e:
                app.logger.exception("analyze/multi: exception during cnn scoring: %s", e)
                c_reasons = [f"cnn exception: {e}"]
                cscore = None
    else:
        c_reasons = ["no image provided"]
        cscore = None

    app.logger.debug("analyze/multi: final cscore=%s, c_reasons=%s", repr(cscore), c_reasons)



    # GNN score with caching
    gscore = None
    g_reasons = []
    if domain:
        if graph_engine is None:
            g_reasons = ["graph engine not initialized"]
            gscore = None
            app.logger.debug("analyze/multi: domain provided but graph_engine is not initialized")
        else:
            # Check cache first
            if domain in gnn_cache:
                gscore = gnn_cache[domain]
                app.logger.debug("Using cached GNN result for domain: %s", domain)
            else:
                try:
                    # graph_engine.predict_node_score now returns 0.0 for unknown domains instead of None
                    pred = graph_engine.predict_node_score(domain)
                    gscore = float(pred) if pred is not None else 0.0
                    gnn_cache[domain] = gscore
                    # Limit cache size
                    if len(gnn_cache) > 200:
                        gnn_cache.clear()
                except Exception as e:
                    app.logger.warning("Graph predict failed: %s", e)
                    g_reasons = [str(e)]
                    gscore = 0.0
    else:
        g_reasons = ["no domain provided"]
        gscore = 0.0

    # Debug log of what we've computed
    app.logger.debug("analyze/multi called: text_len=%d, image_present=%s, domain=%s", len(text or ""), bool(image_b64), domain)
    app.logger.debug("scores so far: tscore=%s, cscore=%s, gscore=%s", tscore, cscore, gscore)

    # --- Advanced AI Agent (ValidationAgent) Logic ---
    agent_reasons = []
    agent_override_score = None
    
    # 1. Major Site Check
    is_major = domain.lower() in MAJOR_SITES
    if is_major:
        agent_override_score = 0.01
        agent_reasons.append("AI Agent: Verified major application site.")

    # 2. TLD Trust Check (Banks, Gov, Edu)
    trust_tlds = [".gov", ".gov.uk", ".ac.uk", ".bank", ".mil", ".edu", ".int"]
    if any(domain.lower().endswith(tld) for tld in trust_tlds):
        agent_override_score = 0.05
        agent_reasons.append(f"AI Agent: High-trust TLD detected ({domain.split('.')[-1]}).")

    # 2.5 Edge Case Threat Scanners (Piracy/Proxies/Torrent)
    # E.g. "pirate bay proxy", "torrent", unblockers
    bad_keywords = ["pirate bay", "thepiratebay", "torrent proxy", "1337x", "yts.", "free movies", "crack download", "unblock proxy"]
    text_lower = text.lower()
    domain_lower = domain.lower()
    
    # Only run threat check if it's not a verified safe site
    if not is_major and agent_override_score is None:
        detected_threat = next((kw for kw in bad_keywords if kw in domain_lower or kw in text_lower), None)
        if detected_threat:
            agent_override_score = 0.95
            agent_reasons.append(f"AI Agent: ALERT - Known malicious/edge-case pattern detected ('{detected_threat}').")

    # 3. Brand Impersonation Detection (CNN Cross-Ref)
    if isinstance(raw_c_res, dict):
        best_brand = str(raw_c_res.get("best_brand", "")).lower()
        best_sim = float(raw_c_res.get("best_sim", 0.0))
        
        if best_sim > 0.85 and best_brand:
            # Check if current domain matches any official domains for detected brand
            official_list = OFFICIAL_DOMAINS.get(best_brand, [])
            domain_is_official = any(o in domain.lower() for o in official_list)
            
            if domain_is_official:
                agent_override_score = 0.02
                agent_reasons.append(f"AI Agent: Visual identity matches official {best_brand} domain.")
            else:
                # CRITICAL: High visual match but unofficial domain
                tscore = max(tscore, 0.95) # force high LLM score
                agent_reasons.append(f"AI Agent: ALERT - Visual identity of '{best_brand.upper()}' detected on unofficial domain!")

    # 4. Domain Age Check (for Banks/High-Value targets)
    if not is_major and agent_override_score is None:
        try:
            w_info = domain_whois_info(domain)
            age = w_info.get("age_days")
            if age and age > 730: # 2 years+
                # Older domains are less likely to be fresh phish kills
                if tscore < 0.6: # If other signals are okay, trust legacy
                    agent_override_score = 0.15
                    agent_reasons.append(f"AI Agent: Established domain (Age: {age} days). Lowering risk.")
        except Exception:
            pass

    # Combine
    final = combine_scores(tscore, cnn_score=cscore, gnn_score=gscore)
    
    # Enrichment (URL details)
    url_info = {"score": 0.0, "reasons": [], "components": {}}
    if domain:
        try:
            w = domain_whois_info(domain)
            url_info["components"] = {"whois": w, "asn": domain_asn_info(domain)}
            if w.get("age_days"): url_info["reasons"].append(f"Domain age: {w['age_days']} days")
        except Exception: pass

    # Apply ValidationAgent Overrides
    if agent_override_score is not None:
        final["score"] = agent_override_score
        final["label"] = "low" if agent_override_score < 0.3 else "high"
        final["badge"] = "🟢" if agent_override_score < 0.2 else "🔴"
        final["risk_label"] = final["label"]
    
    if agent_reasons:
        # Use final.get("reasons", []) as default for safer merging
        final["reasons"] = agent_reasons + final.get("reasons", [])

    # Map labels for dashboard (legacy: high/medium/low vs phish/suspicious/legit)
    label_map = {"phish": "high", "suspicious": "medium", "legit": "low"}
    dashboard_label = label_map.get(final["label"], final["label"])

    final["ok"] = True
    final["aggregate_score"] = final["score"]
    final["label"] = dashboard_label
    final["timestamp"] = datetime.now(IST).isoformat()
    final["text_reasons"] = t_res.get("reasons", [])
    final["cnn_reasons"] = c_reasons
    final["gnn_reasons"] = g_reasons
    final["cnn_score"] = cscore
    final["gnn_score"] = gscore
    final["text_score"] = tscore
    final["url"] = url_info
    final["screenshot_url"] = screenshot_url
    
    # include a field indicating whether the cnn/gnn components were actually run
    final["components_run"] = {"text": True, "cnn": (image_b64 is not None and cnn_scorer is not None), "gnn": (domain != "" and graph_engine is not None)}
    # Include raw component values (allow None)
    final["components_raw"] = {"text": tscore, "cnn": cscore, "gnn": gscore}
    # Persist to aggregate_log.csv
    try:
        agg_path = AGG_LOG
        ts_now = final.get("timestamp") or datetime.now(IST).isoformat()
        txt_val = round(float(tscore), 6)
        cnn_val = "" if cscore is None else round(float(cscore), 6)
        gnn_val = "" if gscore is None else round(float(gscore), 6)
        comb_val = round(float(final.get("score", 0.0)), 6)
        
        # Reasons summary for CSV
        all_r = final.get("reasons", []) + final.get("text_reasons", []) + final.get("cnn_reasons", []) + final.get("gnn_reasons", [])
        reasons_str = "; ".join(str(r) for r in all_r if r)[:1000]
        
        headers = ["timestamp", "url", "domain", "text_score", "cnn_score", "gnn_score", "combined_score", "label", "text_excerpt", "combined_reasons", "screenshot_url"]
        row = [
            ts_now,
            data.get("url") or "",
            domain,
            txt_val,
            cnn_val,
            gnn_val,
            comb_val,
            final["label"],
            (text[:200]).replace("\n", " "),
            reasons_str,
            screenshot_url or ""
        ]
        
        if not os.path.exists(agg_path):
            with open(agg_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(headers)
        with open(agg_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
    except Exception as e:
        app.logger.error("Persistence failed: %s", e)

    return jsonify(final)

@app.route("/graph/reload", methods=["POST"])
def graph_reload():
    if graph_engine is None:
        return jsonify({"ok": False, "error": "GraphEngine not initialized"}), 500
    try:
        AGG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aggregate_log.csv")
        edges = extract_edges_from_aggregate(AGG_PATH)
        graph_engine.build_graph_from_edges(edges)
        graph_engine.compute_node2vec_embeddings(dimensions=64, walk_length=10, num_walks=80)
        return jsonify({"ok": True, "nodes": len(graph_engine.embeddings)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500



# ---------------------------------------------------------------------------
# /analyze/url-quick  — Demo endpoint: score a URL from its string only.
# Zero network calls, zero model inference. Purely heuristic.
# POST { "url": "https://example.com/login?redirect=abc" }
# Returns { ok, score, label, badge, signals, reasons }
# ---------------------------------------------------------------------------
@app.route("/analyze/url-quick", methods=["POST"])
def analyze_url_quick():
    """
    Instant heuristic phishing score derived entirely from the URL string.
    No DNS, no HTTP, no model — safe for demo / offline use.
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
    except Exception:
        data = {}

    raw_url = (data.get("url") or "").strip()
    if not raw_url:
        return jsonify({"ok": False, "error": "url field is required"}), 400

    # Normalise: add scheme if missing so urlparse works
    url_to_parse = raw_url if re.match(r"^https?://", raw_url, re.I) else "http://" + raw_url

    try:
        parsed = urlparse(url_to_parse)
    except Exception as e:
        return jsonify({"ok": False, "error": f"Invalid URL: {e}"}), 400

    scheme   = (parsed.scheme  or "").lower()
    host     = (parsed.hostname or "").lower()
    path     = (parsed.path    or "")
    query    = (parsed.query   or "")
    fragment = (parsed.fragment or "")
    full_url = raw_url

    # ---- Known-safe whitelist (instant 0-score exit) ----------------------
    SAFE_DOMAINS = {
        "google.com", "www.google.com", "youtube.com", "www.youtube.com",
        "github.com", "www.github.com", "microsoft.com", "www.microsoft.com",
        "apple.com", "www.apple.com", "amazon.com", "www.amazon.com",
        "linkedin.com", "www.linkedin.com", "twitter.com", "x.com",
        "instagram.com", "facebook.com", "netflix.com", "wikipedia.org",
        "stackoverflow.com", "reddit.com",
    }
    if host in SAFE_DOMAINS:
        return jsonify({
            "ok": True, "score": 0.02, "label": "low", "badge": "🟢",
            "reasons": ["Verified major/well-known domain."],
            "signals": {"safe_whitelist": True},
            "url": raw_url,
        })

    # ---- Individual signal checks -----------------------------------------
    signals = {}
    reasons = []
    score   = 0.0   # accumulated (capped at 1.0)

    # 1. HTTP (not HTTPS)
    signals["uses_http"] = scheme == "http"
    if signals["uses_http"]:
        score += 0.10
        reasons.append("Uses HTTP instead of HTTPS — connection is unencrypted.")

    # 2. IP address as host
    ip_pattern = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
    signals["ip_as_host"] = bool(ip_pattern.match(host))
    if signals["ip_as_host"]:
        score += 0.25
        reasons.append("Host is a raw IP address — phishing sites often avoid domain names.")

    # 3. URL total length
    url_len = len(full_url)
    signals["url_length"] = url_len
    if url_len > 100:
        inc = min(0.15, (url_len - 100) / 400)
        score += inc
        reasons.append(f"URL is very long ({url_len} chars) — long URLs are used to hide the true destination.")

    # 4. Excessive subdomains
    parts = host.split(".")
    sub_count = max(0, len(parts) - 2)
    signals["subdomain_count"] = sub_count
    if sub_count >= 3:
        score += 0.15
        reasons.append(f"Unusually deep subdomain structure ({sub_count} levels) — common in phishing URLs.")
    elif sub_count == 2:
        score += 0.05
        reasons.append(f"Multiple subdomains detected ({sub_count} levels).")

    # 5. Suspicious keywords in host
    HOST_KEYWORDS = [
        "login", "signin", "secure", "verify", "account", "update",
        "banking", "paypal", "amazon", "apple", "google", "microsoft",
        "ebay", "confirm", "password", "credential", "wallet", "crypto",
        "invoice", "payment", "support", "helpdesk", "service",
    ]
    found_host_kw = [kw for kw in HOST_KEYWORDS if kw in host]
    signals["host_suspicious_keywords"] = found_host_kw
    if found_host_kw:
        inc = min(0.25, len(found_host_kw) * 0.08)
        score += inc
        reasons.append(f"Suspicious keywords in hostname: {', '.join(found_host_kw)}.")

    # 6. Suspicious keywords in path / query
    PATH_KEYWORDS = [
        "login", "signin", "verify", "secure", "account", "password",
        "update", "confirm", "redirect", "token", "credential", "reset",
    ]
    path_query = (path + "?" + query).lower()
    found_path_kw = [kw for kw in PATH_KEYWORDS if kw in path_query]
    signals["path_suspicious_keywords"] = found_path_kw
    if found_path_kw:
        inc = min(0.15, len(found_path_kw) * 0.05)
        score += inc
        reasons.append(f"Suspicious keywords in path/query: {', '.join(found_path_kw)}.")

    # 7. @ symbol in URL (pre-@ part is ignored by browser)
    signals["has_at_symbol"] = "@" in full_url
    if signals["has_at_symbol"]:
        score += 0.20
        reasons.append("URL contains '@' — the part before '@' is ignored; this is a classic phishing trick.")

    # 8. Double slash after host (redirect abuse)
    signals["double_slash_in_path"] = "//" in path
    if signals["double_slash_in_path"]:
        score += 0.10
        reasons.append("Double '//' in path — may be used to confuse redirect parsers.")

    # 9. Hex / percent encoding in host
    signals["hex_encoding_in_host"] = "%" in host
    if signals["hex_encoding_in_host"]:
        score += 0.15
        reasons.append("Percent-encoded characters in host — used to obfuscate the real URL.")

    # 10. Punycode / IDN (internationalized domain — lookalike attack)
    signals["is_punycode"] = host.startswith("xn--") or any(p.startswith("xn--") for p in parts)
    if signals["is_punycode"]:
        score += 0.20
        reasons.append("Punycode (internationalized) domain detected — may be a Unicode lookalike attack.")

    # 11. Risky TLD
    RISKY_TLDS = {
        ".tk", ".ml", ".ga", ".cf", ".gq",   # Freenom free TLDs
        ".xyz", ".top", ".club", ".online", ".site", ".icu",
        ".ru", ".cn", ".th", ".ir",
    }
    tld = "." + parts[-1] if parts else ""
    signals["risky_tld"] = tld in RISKY_TLDS
    if signals["risky_tld"]:
        score += 0.15
        reasons.append(f"High-risk TLD '{tld}' — frequently used in phishing campaigns.")

    # 12. Dash-heavy domain (brand-login.evil.com style)
    dash_count = host.count("-")
    signals["dash_count"] = dash_count
    if dash_count >= 2:
        inc = min(0.15, dash_count * 0.05)
        score += inc
        reasons.append(f"Many hyphens ({dash_count}) in domain — often used to mimic legitimate brands.")

    # 13. Numeric characters in domain (e.g. payp4l.com)
    digits_in_host = sum(c.isdigit() for c in host.replace(".", ""))
    signals["digits_in_domain"] = digits_in_host
    if digits_in_host >= 3:
        score += 0.08
        reasons.append(f"Multiple digits ({digits_in_host}) in domain — may be a typosquatted version of a real brand.")

    # 14. Non-standard port
    port = parsed.port
    signals["non_standard_port"] = port not in (None, 80, 443)
    if signals["non_standard_port"]:
        score += 0.10
        reasons.append(f"Non-standard port ({port}) — legitimate sites rarely use custom ports.")

    # 15. Fragment used (rare in phishing but worth noting for obfuscation)
    signals["has_fragment"] = bool(fragment)

    # 16. Many query parameters (data harvesting)
    param_count = len(query.split("&")) if query else 0
    signals["query_param_count"] = param_count
    if param_count >= 5:
        score += 0.07
        reasons.append(f"Many query parameters ({param_count}) — may be harvesting user data via the URL.")

    # ---- Clamp and label ---------------------------------------------------
    score = round(min(1.0, max(0.0, score)), 4)

    if score < 0.25:
        label, badge = "low",    "🟢"
    elif score < 0.55:
        label, badge = "medium", "🟡"
    else:
        label, badge = "high",   "🔴"

    if not reasons:
        reasons.append("No suspicious signals detected in the URL structure.")

    return jsonify({
        "ok":      True,
        "url":     raw_url,
        "score":   score,
        "label":   label,
        "badge":   badge,
        "reasons": reasons,
        "signals": signals,
        "method":  "url-heuristic-only",
        "note":    "Score based purely on URL structure — no network calls or ML models used.",
    })


if __name__=="__main__":

    port=int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0",port=port,debug=True)
