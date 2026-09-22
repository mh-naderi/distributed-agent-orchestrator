"""
Fetch one web page and return its text.

WHY THIS AGENT EXISTS. Until now the system could find pages and never read
one. `search_web` returns a title, a snippet and a URL; `retrieve` returns
stored fragments. The documented fabrication path is answers built from "real
documents about a different subject", and thin snippets are what made that easy:
the model had nothing substantive to read, so the gap between "these results are
about something else" and "these results answer the question" had to be guessed.

WHAT MAKES IT MORE THAN urlopen. A service that fetches any URL a model asks for
is a proxy with an LLM choosing the targets, and it runs INSIDE the cluster. The
rules below exist because of what a pod can reach:

  - the Kubernetes API server and every ClusterIP service
  - the node itself on 127.0.0.1
  - on a cloud node, the metadata endpoint at 169.254.169.254, which hands out
    credentials to anybody who asks

So the address is resolved and checked BEFORE the request, every redirect is
re-checked the same way, and anything that is not a public address is refused
with a reason. This is not hypothetical hardening: the eval's own corpus
contains text the model composed from a question, and a model that will invent
a foundation will invent a URL.

WHAT IT DOES NOT DO, stated in the output the way analyze_code states its
limits: it does not run JavaScript, so a page whose text is rendered client-side
comes back nearly empty; it reads only the first bytes of a large page; and it
extracts text by stripping markup rather than by understanding layout, so
navigation and boilerplate arrive mixed in with the article.
"""

import html
import ipaddress
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

# A page is read for its prose. These bound what the agent will pull into the
# orchestrator's context window, which is 4096 tokens for the whole run.
MAX_BYTES = 400_000
MAX_CHARS = 8_000
TIMEOUT_SECONDS = 10.0
MAX_REDIRECTS = 3

ALLOWED_SCHEMES = ("http", "https")

# text/html and text/plain are what a page is. A PDF or an image would arrive as
# bytes this agent cannot turn into prose, and saying so is better than handing
# back mojibake that reads like a failed extraction.
ALLOWED_CONTENT_TYPES = ("text/html", "text/plain", "application/xhtml+xml")

_SCRIPT_OR_STYLE = re.compile(
    r"<(script|style|noscript|template)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL
)
_TAG = re.compile(r"<[^>]+>")
_BLANKS = re.compile(r"\n{3,}")
_SPACES = re.compile(r"[ \t\x0b\f\r]+")
# Tags whose boundaries are paragraph breaks in the text, so stripping markup
# does not run three sections into one sentence.
_BLOCK_END = re.compile(
    r"</(p|div|section|article|h[1-6]|li|tr|blockquote|pre)\s*>", re.IGNORECASE
)
_LINE_BREAK = re.compile(r"<br\s*/?>", re.IGNORECASE)


class FetchRefused(Exception):
    """The request was not made, and why. Never a partial or silent result."""


@dataclass
class Page:
    url: str
    text: str
    truncated: bool
    content_type: str


def _is_public(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    # is_global is not enough on its own: it is False for private ranges but
    # link-local (169.254.0.0/16, where cloud metadata lives), loopback and
    # multicast each need to be out regardless of how is_global treats them on
    # a given Python version.
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def check_url(url: str, resolve=None) -> str:
    """
    Refuse anything that is not a public http(s) address, and say why.

    `resolve` is injected so tests can state what a hostname resolves to without
    touching DNS. It returns the list of addresses a connection might use; ALL of
    them must be public, because a name that resolves to both a public and a
    private address would otherwise be a coin toss.
    """
    resolve = resolve or _resolve
    parsed = urllib.parse.urlparse(url)

    if parsed.scheme not in ALLOWED_SCHEMES:
        raise FetchRefused(
            f"refused: {parsed.scheme or 'no'} scheme. This tool fetches http and "
            "https pages only."
        )
    if not parsed.hostname:
        raise FetchRefused("refused: no host in that URL.")

    try:
        addresses = resolve(parsed.hostname)
    except OSError as exc:
        raise FetchRefused(f"refused: {parsed.hostname} does not resolve ({exc}).")

    if not addresses:
        raise FetchRefused(f"refused: {parsed.hostname} does not resolve.")

    for address in addresses:
        if not _is_public(address):
            raise FetchRefused(
                f"refused: {parsed.hostname} resolves to {address}, which is not a "
                "public address. This agent runs inside the cluster, where private "
                "addresses reach other services, the node and cloud metadata."
            )
    return url


def _resolve(hostname: str) -> list[str]:
    infos = socket.getaddrinfo(hostname, None)
    return sorted({info[4][0] for info in infos})


def extract_text(body: str, content_type: str) -> str:
    """
    Markup to prose, approximately and on purpose.

    Approximate is stated rather than hidden: block ends become paragraph
    breaks, tags are dropped, entities are unescaped. What survives is the page's
    words in their order, including navigation and boilerplate. A proper
    readability pass would be a dependency and a second thing to be wrong.
    """
    if "html" not in content_type:
        return body.strip()

    text = _SCRIPT_OR_STYLE.sub(" ", body)
    text = _LINE_BREAK.sub("\n", text)
    text = _BLOCK_END.sub("\n\n", text)
    text = _TAG.sub(" ", text)
    text = html.unescape(text)
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _BLANKS.sub("\n\n", text).strip()


def fetch(url: str, opener=None, resolve=None) -> Page:
    """
    One page, with every redirect re-checked.

    Redirects are followed by hand because the safety check has to run again on
    each hop. urllib's default opener follows them itself, which would let a
    public URL redirect to 169.254.169.254 and hand back the result of a request
    nobody approved.
    """
    opener = opener or _open
    current = check_url(url, resolve)

    for _ in range(MAX_REDIRECTS + 1):
        status, headers, body, location = opener(current, TIMEOUT_SECONDS, MAX_BYTES)

        if status in (301, 302, 303, 307, 308) and location:
            current = check_url(urllib.parse.urljoin(current, location), resolve)
            continue

        if status != 200:
            raise FetchRefused(f"the server answered {status} for {current}.")

        content_type = (headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if content_type and content_type not in ALLOWED_CONTENT_TYPES:
            raise FetchRefused(
                f"refused: {current} is {content_type}, which this tool cannot turn "
                "into text. It reads HTML and plain text."
            )

        text = extract_text(body, content_type or "text/html")
        truncated = len(text) > MAX_CHARS
        return Page(
            url=current,
            text=text[:MAX_CHARS],
            truncated=truncated or len(body.encode("utf-8", "ignore")) >= MAX_BYTES,
            content_type=content_type or "unknown",
        )

    raise FetchRefused(f"refused: more than {MAX_REDIRECTS} redirects starting at {url}.")


def _open(url: str, timeout: float, max_bytes: int):
    """The one place this agent touches the network."""

    class _NoRedirects(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None  # handled in fetch(), so each hop is re-checked

    opener = urllib.request.build_opener(_NoRedirects)
    request = urllib.request.Request(
        url,
        headers={
            # Identifying the caller is the polite half of scraping; the other
            # half is MIN_SECONDS_BETWEEN_SEARCHES in the research agent.
            "User-Agent": "distributed-agent-orchestrator/1.0 (reader agent)",
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9",
        },
    )
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read(max_bytes)
            return response.status, dict(response.headers), _decode(raw, response.headers), None
    except urllib.error.HTTPError as exc:
        location = exc.headers.get("Location") if exc.headers else None
        return exc.code, dict(exc.headers or {}), "", location
    except urllib.error.URLError as exc:
        raise FetchRefused(f"could not reach {url}: {exc.reason}")


def _decode(raw: bytes, headers) -> str:
    charset = None
    if headers:
        charset = headers.get_content_charset() if hasattr(headers, "get_content_charset") else None
    return raw.decode(charset or "utf-8", errors="replace")
