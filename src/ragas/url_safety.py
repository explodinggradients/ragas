from __future__ import annotations

import ipaddress
import logging
import socket
import typing as t
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

ALLOWED_URL_SCHEMES = {"http", "https"}
DISALLOWED_IP_CHECKS = {"is_loopback", "is_private", "is_link_local", "is_reserved"}


def prepare_safe_http_url(
    url: str, *, allow_internal_targets: bool = False
) -> t.Optional[str]:
    """
    Normalize a URL the same way requests will send it, then block internal targets.

    Returning the prepared URL keeps the validation aligned with the actual request
    target, which avoids parser mismatches like backslash-based SSRF bypasses.
    """
    try:
        prepared_url = requests.Request("GET", url).prepare().url
    except requests.exceptions.InvalidURL as exc:
        logger.error(f"Invalid URL '{url}': {exc}")
        return None
    except Exception as exc:
        logger.error(f"Unexpected error preparing URL '{url}': {exc}")
        return None

    parsed_url = urlparse(prepared_url)
    if parsed_url.scheme not in ALLOWED_URL_SCHEMES:
        logger.error(f"URL '{url}' uses unsupported scheme '{parsed_url.scheme}'.")
        return None

    if not parsed_url.hostname:
        logger.error(f"Could not extract hostname from prepared URL '{prepared_url}'.")
        return None

    if not allow_internal_targets and not _is_safe_url_target(parsed_url.hostname):
        return None

    return prepared_url


def _is_safe_url_target(url_hostname: str) -> bool:
    """
    Resolve a hostname and reject IPs that point at internal networks.
    """
    try:
        addrinfo_results = socket.getaddrinfo(
            url_hostname, None, family=socket.AF_UNSPEC
        )
        if not addrinfo_results:
            logger.error(
                f"SSRF check: DNS resolution failed for hostname '{url_hostname}' (no results)"
            )
            return False

        for _, _, _, _, sockaddr in addrinfo_results:
            ip_address_str = sockaddr[0]
            try:
                ip = ipaddress.ip_address(ip_address_str)
            except ValueError as exc:
                logger.error(
                    f"SSRF check: Error parsing resolved IP address '{ip_address_str}' "
                    f"for hostname '{url_hostname}': {exc}"
                )
                return False

            for check_name in DISALLOWED_IP_CHECKS:
                if getattr(ip, check_name, False):
                    logger.error(
                        f"SSRF check: Hostname '{url_hostname}' resolved to disallowed "
                        f"IP '{ip_address_str}' ({check_name}=True). Blocking request."
                    )
                    return False

        return True
    except socket.gaierror as exc:
        logger.error(
            f"SSRF check: DNS resolution error for hostname '{url_hostname}': {exc}"
        )
        return False
    except Exception as exc:
        logger.error(
            f"SSRF check: Unexpected error checking hostname '{url_hostname}': {exc}"
        )
        return False
