# backend/domain_info.py
import whois
import socket
from ipwhois import IPWhois
from urllib.parse import urlparse
from datetime import datetime

def domain_whois_info(domain: str):
    """
    Perform WHOIS lookup for a domain, return basic info like creation date, registrar, and age in days.
    """
    try:
        w = whois.whois(domain)
        registrar = w.registrar
        creation_date = w.creation_date
        expiration_date = w.expiration_date

        # Normalize if multiple dates are returned
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        if isinstance(expiration_date, list):
            expiration_date = expiration_date[0]

        age_days = None
        if creation_date and isinstance(creation_date, datetime):
            # Handle timezone-aware vs timezone-naive datetime comparison
            now = datetime.utcnow()
            if creation_date.tzinfo is not None and now.tzinfo is None:
                # creation_date is timezone-aware, now is naive
                now = now.replace(tzinfo=creation_date.tzinfo)
            elif creation_date.tzinfo is None and now.tzinfo is not None:
                # creation_date is naive, now is timezone-aware
                creation_date = creation_date.replace(tzinfo=now.tzinfo)
            age_days = (now - creation_date).days

        return {
            "registrar": registrar,
            "creation_date": str(creation_date) if creation_date else None,
            "expiration_date": str(expiration_date) if expiration_date else None,
            "age_days": age_days
        }
    except Exception as e:
        return {"error": f"whois_error: {str(e)}"}

def domain_asn_info(domain: str):
    """
    Resolve domain to IP and fetch ASN info (org, country, ASN).
    """
    try:
        ip = socket.gethostbyname(domain)
        obj = IPWhois(ip)
        res = obj.lookup_rdap(asn_methods=["whois"])
        return {
            "ip": ip,
            "asn": res.get("asn"),
            "asn_cidr": res.get("asn_cidr"),
            "asn_country": res.get("asn_country_code"),
            "asn_org": res.get("asn_description")
        }
    except Exception as e:
        return {"error": f"asn_error: {str(e)}"}
