"""Shared HTTP helpers.

All three data sources are public services that occasionally rate-limit or time
out. One retry helper with backoff keeps that handling in a single place.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

import requests

LOG = logging.getLogger(__name__)

USER_AGENT = "MaptileConverter/1.0 (offline 3D map pipeline)"

# OGC services answer errors with an XML ServiceExceptionReport under HTTP 200,
# so the status code alone never proves a request succeeded.
_XML_SNIFF = (b"<?xml", b"<Service", b"<ows:", b"<ExceptionReport")


class ServiceError(RuntimeError):
    """A service answered, but with an error document instead of data."""


def looks_like_xml(payload: bytes) -> bool:
    head = payload.lstrip()[:64]
    return any(head.startswith(marker) for marker in _XML_SNIFF)


def extract_service_exception(payload: bytes) -> str:
    """Pull the readable message out of an OGC exception document."""
    import re

    text = payload.decode("utf-8", errors="replace")
    # The tag name has to end at the match, otherwise <ServiceException...>
    # also matches the enclosing <ServiceExceptionReport ...> and the message
    # comes back wrapped in a stray tag.
    for pattern in (
        r"<ServiceException(?:\s[^>]*)?>(.*?)</ServiceException>",
        r"<ows:ExceptionText(?:\s[^>]*)?>(.*?)</ows:ExceptionText>",
        r"<ExceptionText(?:\s[^>]*)?>(.*?)</ExceptionText>",
    ):
        match = re.search(pattern, text, re.S)
        if match:
            return " ".join(match.group(1).split())
    return " ".join(text.split())[:400]


def get_with_retry(
    url: str,
    params: dict[str, Any] | None = None,
    *,
    timeout: float = 120.0,
    max_retries: int = 4,
    session: requests.Session | None = None,
    expect_binary: bool = False,
    description: str = "request",
) -> requests.Response:
    """GET with exponential backoff.

    Retries on transport errors, on 5xx and 429, and — when `expect_binary` is
    set — on OGC exception documents returned under HTTP 200.
    """
    http = session or requests
    delay = 2.0
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            response = http.get(
                url,
                params=params,
                timeout=timeout,
                headers={"User-Agent": USER_AGENT},
            )
        except requests.RequestException as exc:
            last_error = exc
            LOG.warning(
                "%s attempt %d/%d failed: %s", description, attempt, max_retries, exc
            )
        else:
            if response.status_code in (429, 500, 502, 503, 504):
                last_error = ServiceError(
                    f"{description} returned HTTP {response.status_code}"
                )
                LOG.warning(
                    "%s attempt %d/%d got HTTP %d",
                    description,
                    attempt,
                    max_retries,
                    response.status_code,
                )
            elif not response.ok:
                # 4xx other than 429 will not fix itself; fail immediately.
                raise ServiceError(
                    f"{description} failed with HTTP {response.status_code}: "
                    f"{response.text[:400]}"
                )
            elif expect_binary and looks_like_xml(response.content):
                message = extract_service_exception(response.content)
                # A service exception is a rejected request, not a flaky one.
                raise ServiceError(f"{description} was refused: {message}")
            else:
                return response

        if attempt < max_retries:
            LOG.info("retrying %s in %.0fs", description, delay)
            time.sleep(delay)
            delay *= 2

    raise ServiceError(
        f"{description} failed after {max_retries} attempts"
    ) from last_error


def retry_call(
    func: Callable[[], Any],
    *,
    max_retries: int = 4,
    description: str = "call",
) -> Any:
    """Run `func` with the same backoff policy as :func:`get_with_retry`."""
    delay = 2.0
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001 - retry policy is deliberate
            last_error = exc
            LOG.warning(
                "%s attempt %d/%d failed: %s", description, attempt, max_retries, exc
            )
            if attempt < max_retries:
                time.sleep(delay)
                delay *= 2
    raise ServiceError(f"{description} failed after {max_retries} attempts") from last_error


__all__ = [
    "ServiceError",
    "USER_AGENT",
    "extract_service_exception",
    "get_with_retry",
    "looks_like_xml",
    "retry_call",
]
