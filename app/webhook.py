#!/usr/bin/env python3
"""
Push Webhook Delivery Engine for whisper-k8s.

Provides safe, reliable webhook notification delivery with:
1. Strict SSRF validation (blocking private IPs, loopback, link-local, cloud metadata).
2. Exponential backoff retries on network failures and 5xx server errors.
3. Clean standard library implementation with zero external dependencies.
"""

import os
import time
import json
import socket
import logging
import ipaddress
import urllib.parse
import urllib.request
import urllib.error
from typing import Dict, Any, Optional

try:
    import metrics  # type: ignore
except ImportError:
    try:
        from app import metrics  # type: ignore
    except ImportError:
        metrics = None  # type: ignore

logger = logging.getLogger(__name__)


def validate_webhook_url(url: str) -> None:
    """
    Validate that a webhook URL is safe.

    Protects against SSRF by:
    - Enforcing http or https schemes.
    - Resolving hostname and rejecting private, loopback, link-local,
      multicast, reserved, and cloud metadata (169.254.169.254) addresses.
    - Can be bypassed in local testing by setting ALLOW_LOCAL_URLS=true.
    """
    if not url or not isinstance(url, str):
        raise ValueError("Webhook URL must be a non-empty string")

    parsed = urllib.parse.urlparse(url)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError(f"Disallowed URL scheme: '{parsed.scheme}'. Only http and https are allowed.")

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("Invalid URL: missing hostname")

    if os.getenv("ALLOW_LOCAL_URLS", "false").strip().lower() in {"true", "1", "yes"}:
        return

    # Check for localhost / loopback aliases directly
    if hostname.lower() in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Access to loopback addresses is forbidden (SSRF protection)")

    try:
        addr_info = socket.getaddrinfo(hostname, None)
    except socket.gaierror as e:
        raise ValueError(f"Cannot resolve hostname '{hostname}': {e}")

    for info in addr_info:
        ip_str = info[4][0]
        try:
            ip = ipaddress.ip_address(ip_str)
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_reserved
                or ip.is_multicast
                or str(ip) == "169.254.169.254"
            ):
                raise ValueError(
                    f"Disallowed remote target: IP '{ip_str}' is private or reserved (SSRF protection)."
                )
        except ValueError as e:
            if "is private or reserved" in str(e):
                raise
            raise ValueError(f"Invalid resolved IP address: '{ip_str}'")


def send_webhook(
    url: str,
    payload: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    timeout: float = 10.0,
) -> bool:
    """
    Dispatch an HTTP POST webhook with JSON payload and exponential backoff.

    Parameters
    ----------
    url : str
        Target webhook endpoint.
    payload : dict
        Arbitrary JSON-serializable dictionary.
    headers : dict, optional
        Additional HTTP headers to include (e.g. Authorization, X-Secret).
    max_retries : int
        Maximum delivery attempts (default 3).
    retry_delay : float
        Initial retry backoff delay in seconds (default 1.0s, doubled each retry).
    timeout : float
        Socket timeout in seconds (default 10.0s).

    Returns
    -------
    bool
        True if the webhook was delivered and received a 2xx response, False otherwise.
    """
    try:
        validate_webhook_url(url)
    except Exception as e:
        logger.error("[webhook] SSRF or URL validation failed for %s: %s", url, e)
        if metrics and hasattr(metrics, "WEBHOOKS_DISPATCHED"):
            metrics.WEBHOOKS_DISPATCHED.inc(1.0, status="failure")
        return False

    body = json.dumps(payload).encode("utf-8")
    req_headers = {
        "Content-Type": "application/json",
        "User-Agent": "whisper-k8s-webhook/1.0",
    }
    if headers:
        req_headers.update(headers)

    delay = retry_delay
    for attempt in range(1, max_retries + 1):
        try:
            logger.info("[webhook] Sending webhook attempt %d/%d to %s", attempt, max_retries, url)
            req = urllib.request.Request(url, data=body, headers=req_headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status_code = resp.status if hasattr(resp, "status") else resp.getcode()
                if 200 <= status_code < 300:
                    logger.info("[webhook] Delivery successful (HTTP %d)", status_code)
                    if metrics and hasattr(metrics, "WEBHOOKS_DISPATCHED"):
                        metrics.WEBHOOKS_DISPATCHED.inc(1.0, status="success")
                    return True
                elif 400 <= status_code < 500:
                    logger.warning("[webhook] Delivery rejected with client error HTTP %d", status_code)
                    if metrics and hasattr(metrics, "WEBHOOKS_DISPATCHED"):
                        metrics.WEBHOOKS_DISPATCHED.inc(1.0, status="failure")
                    return False
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500:
                logger.warning("[webhook] Client error HTTP %d from %s, will not retry", e.code, url)
                if metrics and hasattr(metrics, "WEBHOOKS_DISPATCHED"):
                    metrics.WEBHOOKS_DISPATCHED.inc(1.0, status="failure")
                return False
            logger.warning("[webhook] Server error HTTP %d from %s (attempt %d/%d)", e.code, url, attempt, max_retries)
        except Exception as e:
            logger.warning("[webhook] Error dispatching webhook to %s (attempt %d/%d): %s", url, attempt, max_retries, e)

        if attempt < max_retries:
            time.sleep(delay)
            delay *= 2

    logger.error("[webhook] Failed to deliver webhook to %s after %d attempts", url, max_retries)
    if metrics and hasattr(metrics, "WEBHOOKS_DISPATCHED"):
        metrics.WEBHOOKS_DISPATCHED.inc(1.0, status="failure")
    return False
