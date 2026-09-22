"""
The reader agent's safety rules and text extraction.

No network: DNS resolution and the HTTP call are both injected, because what is
under test is what this agent REFUSES, and a test that needs the internet to
prove a refusal would be skipped exactly when it matters.

WHY THESE RULES. This agent fetches URLs an LLM chose, from inside the cluster.
A pod can reach the Kubernetes API, every ClusterIP service, the node on
localhost, and on a cloud node the metadata endpoint that hands out credentials.
The model that picks the URLs is the same one this project has watched invent a
foundation, a report title and a source label.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents" / "reader_agent"))

import fetch  # noqa: E402

PUBLIC = ["93.184.216.34"]


def resolving_to(*addresses):
    return lambda hostname: list(addresses)


# ---------------------------------------------------------------------------
# What it refuses
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "address, what",
    [
        ("127.0.0.1", "the node itself"),
        ("10.96.0.1", "the Kubernetes API service"),
        ("10.244.1.7", "a pod address"),
        ("192.168.1.10", "a LAN host"),
        ("172.16.5.4", "a private range"),
        ("169.254.169.254", "cloud metadata"),
        ("::1", "loopback over IPv6"),
        ("fd00::1", "a unique-local IPv6 address"),
    ],
)
def test_a_private_address_is_refused(address, what):
    with pytest.raises(fetch.FetchRefused) as refused:
        fetch.check_url("http://anything.example", resolve=resolving_to(address))

    message = str(refused.value)
    assert address in message, f"the refusal must name the address ({what})"
    assert "not a public address" in message


def test_a_name_that_resolves_to_both_is_refused():
    """
    A host with one public and one private address is a coin toss at connect
    time. The safe reading of "might connect to either" is to refuse.
    """
    with pytest.raises(fetch.FetchRefused):
        fetch.check_url("http://split.example", resolve=resolving_to("93.184.216.34", "10.0.0.5"))


@pytest.mark.parametrize("url", ["file:///etc/passwd", "gopher://x/1", "ftp://host/f"])
def test_only_http_and_https_are_fetched(url):
    with pytest.raises(fetch.FetchRefused, match="scheme"):
        fetch.check_url(url, resolve=resolving_to(*PUBLIC))


def test_a_host_that_does_not_resolve_is_refused_rather_than_attempted():
    def boom(hostname):
        raise OSError("Name or service not known")

    with pytest.raises(fetch.FetchRefused, match="does not resolve"):
        fetch.check_url("http://nope.example", resolve=boom)


def test_a_public_address_passes():
    assert fetch.check_url("https://example.com/a", resolve=resolving_to(*PUBLIC))


# ---------------------------------------------------------------------------
# Redirects are re-checked, not trusted
# ---------------------------------------------------------------------------


def opener_script(*steps):
    """Each step is (status, headers, body, location), returned in order."""
    calls = []

    def opener(url, timeout, max_bytes):
        calls.append(url)
        return steps[min(len(calls) - 1, len(steps) - 1)]

    opener.calls = calls
    return opener


def test_a_redirect_to_a_private_address_is_refused():
    """
    The attack this design is for: a public URL that redirects inward. urllib's
    default opener would follow it before anything could object.
    """
    opener = opener_script(
        (302, {}, "", "http://metadata.internal/latest/meta-data/"),
        (200, {"Content-Type": "text/html"}, "<p>secrets</p>", None),
    )

    def resolve(hostname):
        return ["169.254.169.254"] if hostname == "metadata.internal" else PUBLIC

    with pytest.raises(fetch.FetchRefused, match="169.254.169.254"):
        fetch.fetch("http://start.example", opener=opener, resolve=resolve)

    assert opener.calls == ["http://start.example"], "the redirect must not be followed"


def test_a_redirect_to_a_public_address_is_followed():
    opener = opener_script(
        (301, {}, "", "https://elsewhere.example/page"),
        (200, {"Content-Type": "text/html"}, "<p>hello</p>", None),
    )

    page = fetch.fetch("http://start.example", opener=opener, resolve=resolving_to(*PUBLIC))

    assert page.url == "https://elsewhere.example/page"
    assert "hello" in page.text


def test_a_redirect_loop_stops():
    opener = opener_script((302, {}, "", "http://round.example/again"))

    with pytest.raises(fetch.FetchRefused, match="redirects"):
        fetch.fetch("http://round.example", opener=opener, resolve=resolving_to(*PUBLIC))


# ---------------------------------------------------------------------------
# What comes back
# ---------------------------------------------------------------------------


def test_an_error_status_is_reported_not_swallowed():
    opener = opener_script((404, {}, "", None))

    with pytest.raises(fetch.FetchRefused, match="404"):
        fetch.fetch("http://example.com/missing", opener=opener, resolve=resolving_to(*PUBLIC))


def test_a_pdf_is_refused_rather_than_decoded_into_noise():
    opener = opener_script((200, {"Content-Type": "application/pdf"}, "%PDF-1.7 ...", None))

    with pytest.raises(fetch.FetchRefused, match="application/pdf"):
        fetch.fetch("http://example.com/a.pdf", opener=opener, resolve=resolving_to(*PUBLIC))


def test_long_pages_are_cut_and_say_so():
    body = "<p>" + ("word " * 5000) + "</p>"
    opener = opener_script((200, {"Content-Type": "text/html"}, body, None))

    page = fetch.fetch("http://example.com/long", opener=opener, resolve=resolving_to(*PUBLIC))

    assert len(page.text) == fetch.MAX_CHARS
    assert page.truncated is True


def test_a_short_page_is_not_marked_truncated():
    opener = opener_script((200, {"Content-Type": "text/html"}, "<p>short</p>", None))

    page = fetch.fetch("http://example.com/s", opener=opener, resolve=resolving_to(*PUBLIC))

    assert page.truncated is False


# ---------------------------------------------------------------------------
# Markup to prose
# ---------------------------------------------------------------------------


def test_script_and_style_contents_are_dropped():
    body = """
    <html><head><style>.a { color: red }</style>
    <script>var tracking = "do not read me";</script></head>
    <body><p>The actual sentence.</p></body></html>
    """

    text = fetch.extract_text(body, "text/html")

    assert "The actual sentence." in text
    assert "tracking" not in text and "color: red" not in text


def test_block_boundaries_become_paragraph_breaks():
    """Without this, the last word of a heading joins the first of a paragraph."""
    text = fetch.extract_text("<h1>Title</h1><p>Body text.</p>", "text/html")

    assert "Title\n\nBody text." in text


def test_entities_are_unescaped():
    text = fetch.extract_text("<p>Tom &amp; Jerry &lt;3</p>", "text/html")

    assert "Tom & Jerry <3" in text


def test_plain_text_is_left_alone():
    text = fetch.extract_text("a < b and 2 > 1", "text/plain")

    assert text == "a < b and 2 > 1"
