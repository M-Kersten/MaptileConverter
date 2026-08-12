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

# Connect and read are bounded separately. A read can legitimately take minutes
# — a 12000 px aerial mosaic or a 2000x2000 AHN tile is a lot of bytes — but a
# TCP handshake that has not completed in ten seconds is not going to. Sharing
# one timeout between them means a service that is simply down burns the whole
# read budget on every attempt: at 180 s and four attempts that is twelve
# minutes to discover nobody is listening.
CONNECT_TIMEOUT_S = 10.0

# OGC services answer errors with an XML ServiceExceptionReport under HTTP 200,
# so the status code alone never proves a request succeeded.
_XML_SNIFF = (b"<?xml", b"<Service", b"<ows:", b"<ExceptionReport")


class ServiceError(RuntimeError):
    """A service answered, but with an error document instead of data."""


class ServiceUnreachable(ServiceError):
    """Nothing answered at all: the host is down, or the network is."""


def host_of(url: str) -> str:
    from urllib.parse import urlparse

    return urlparse(url).netloc or url


def is_unreachable(exc: BaseException) -> bool:
    """True when the request never reached a server.

    Worth separating from every other failure: it means the run cannot proceed
    for reasons that have nothing to do with the request or the settings, and
    it is almost always an outage rather than anything the caller can fix.
    """
    return isinstance(exc, (requests.ConnectionError, requests.ConnectTimeout))


def short_error(exc: BaseException) -> str:
    """One readable line instead of urllib3's nested repr.

    The raw form buries the cause in three wrapped exceptions and a connection
    object address, which tells the reader nothing.
    """
    text = str(exc)
    # Ordered most specific first: urllib3 nests these, so a read timeout also
    # mentions the pool and a connect timeout also mentions "Max retries".
    for needle, plain in (
        ("NameResolutionError", "could not resolve the hostname"),
        ("ConnectTimeoutError", "connection timed out"),
        ("Connection timed out", "connection timed out"),
        ("NewConnectionError", "could not open a connection"),
        ("Connection reset by peer", "connection reset by the server"),
        ("ConnectionResetError", "connection reset by the server"),
        ("Read timed out", "connected, but the server never replied"),
        ("ReadTimeoutError", "connected, but the server never replied"),
        ("SSLError", "TLS handshake failed"),
        ("ProxyError", "the proxy refused the connection"),
    ):
        if needle in text:
            return plain
    return text if len(text) < 160 else text[:157] + "..."


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
                timeout=(min(CONNECT_TIMEOUT_S, timeout), timeout),
                headers={"User-Agent": USER_AGENT},
            )
        except requests.RequestException as exc:
            last_error = exc
            # Retries are kept even for connection errors: a dropped link comes
            # back, and with connect bounded separately an attempt now costs
            # seconds rather than the full read budget.
            LOG.warning(
                "%s attempt %d/%d failed: %s",
                description,
                attempt,
                max_retries,
                short_error(exc),
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
                #
                # An OGC service puts the reason in an ExceptionText buried
                # under about 400 characters of XML preamble, so printing the
                # first 400 characters of the body reliably cut off the one
                # sentence worth reading. A 4000 px AHN request reported
                # "...<ows:ExceptionText>msWCS" and stopped there, hiding that
                # the service caps a coverage at 4000 px.
                detail = (
                    extract_service_exception(response.content)
                    if looks_like_xml(response.content)
                    else response.text[:400]
                )
                raise ServiceError(
                    f"{description} failed with HTTP {response.status_code}: "
                    f"{detail}"
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

    if is_unreachable(last_error):
        raise ServiceUnreachable(
            f"{host_of(url)} is unreachable ({short_error(last_error)}). "
            f"Nothing answered after {max_retries} attempts, so this is an "
            f"outage at their end or a network problem here, not something "
            f"the settings can fix."
        ) from last_error

    raise ServiceError(
        f"{description} failed after {max_retries} attempts: "
        f"{short_error(last_error) if last_error else 'no detail'}"
    ) from last_error


def check_reachable(url: str, timeout: float = 8.0) -> tuple[bool, str]:
    """Can we open a connection to this service at all?

    Only connectivity is tested, not correctness: any HTTP reply, including a
    404 or a 400, proves the host is up and answering, which is all the caller
    needs to know before committing to a long run.
    """
    try:
        response = requests.get(
            url,
            timeout=(min(CONNECT_TIMEOUT_S, timeout), timeout),
            headers={"User-Agent": USER_AGENT},
            stream=True,  # headers are enough; do not pull the body
        )
        response.close()
        return True, f"HTTP {response.status_code}"
    except requests.RequestException as exc:
        return False, short_error(exc)


def preflight(services: dict[str, str], timeout: float = 8.0) -> list[str]:
    """Check every service a run needs, and report the ones that are down.

    The buildings stage runs fifth, after several minutes of terrain, imagery
    and BGT work. Discovering there that 3DBAG is offline wastes all of it, so
    the services are checked first, which costs a couple of seconds.
    """
    down: list[str] = []
    for label, url in services.items():
        ok, detail = check_reachable(url, timeout)
        if ok:
            LOG.info("  %-28s reachable (%s)", label, detail)
        else:
            LOG.error("  %-28s UNREACHABLE (%s)", label, detail)
            down.append(f"{label} at {host_of(url)}: {detail}")
    return down


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
    "CONNECT_TIMEOUT_S",
    "ServiceError",
    "ServiceUnreachable",
    "USER_AGENT",
    "check_reachable",
    "extract_service_exception",
    "get_with_retry",
    "host_of",
    "is_unreachable",
    "looks_like_xml",
    "preflight",
    "retry_call",
    "short_error",
]
