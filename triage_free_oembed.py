"""
X Bookmarks Triage (Free, oEmbed) — fallback using oEmbed API (no auth) + Ollama.

Use this if the syndication API (triage_free.py) stops working.
oEmbed returns less data (no metrics, text only) but is very stable.

Requires:
  - Ollama running locally with a model pulled (default: llama3.2)

Usage:
  uv run triage_free_oembed.py
  uv run triage_free_oembed.py --model qwen3:14b
  uv run triage_free_oembed.py --output results.json
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import ollama
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BOOKMARKS_PATH = Path(__file__).parent / "bookmarks.json"
DEFAULT_MODEL = "llama3.2"
OUTPUT_PATH = Path(__file__).parent / "triage_results_oembed.json"
OEMBED_URL = "https://publish.twitter.com/oembed"
FETCH_DELAY = 0.35  # seconds between requests to avoid rate limiting

CATEGORIES = [
    "Technical Article / Tutorial",
    "AI / ML Research & News",
    "Career & Life Advice",
    "Product Launch / Announcement",
    "Open Source / Developer Tool",
    "Design & UI/UX",
    "Business & Startup",
    "Humor / Meme",
    "Crypto / Web3",
    "Health & Wellness",
    "News / Current Events",
    "Other",
]

SYSTEM_PROMPT = f"""\
You are a bookmark triage assistant. You will be given a tweet and must:

1. **Categorize** it into exactly ONE of these categories:
{chr(10).join(f"   - {c}" for c in CATEGORIES)}

2. **Flag as high priority** if the tweet contains genuinely valuable information:
   - High quality articles, tutorials, or deep technical content
   - Insightful advice (career, life, business) that is actionable and thoughtful
   - Important announcements or research breakthroughs
   - Posts with uniquely useful information worth revisiting
   Do NOT flag memes, basic product announcements, low-effort threads, or generic motivational content.

3. **Write a 1-sentence summary** of the tweet's value.

Respond ONLY with valid JSON (no markdown fences):
{{"category": "...", "high_priority": true/false, "summary": "..."}}
"""


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class TweetData:
    url: str
    text: str = ""
    author_name: str = ""
    author_username: str = ""
    fetch_error: str | None = None


@dataclass
class TriageResult:
    tweet: TweetData
    category: str = "Unknown"
    high_priority: bool = False
    summary: str = ""
    llm_error: str | None = None


@dataclass
class TriageState:
    total: int = 0
    processed: int = 0
    fetched: int = 0
    fetch_errors: int = 0
    high_priority_count: int = 0
    current_tweet: str = ""
    current_author: str = ""
    current_phase: str = "Initializing..."
    results: list[TriageResult] = field(default_factory=list)
    category_counts: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# oEmbed fetching
# ---------------------------------------------------------------------------


def _resolve_tco_link(client: httpx.Client, tco_url: str) -> str:
    """Resolve a t.co shortened URL to its final destination via HEAD request."""
    try:
        resp = client.head(tco_url, follow_redirects=True, timeout=10.0)
        final_url = str(resp.url)
        if final_url != tco_url:
            return final_url
    except Exception:
        pass
    return tco_url


def _is_only_links(text: str) -> bool:
    """Check if the tweet text is only URLs with no real content."""
    stripped = re.sub(r"https?://\S+", "", text).strip()
    return len(stripped) == 0


def _extract_text_from_oembed_html(html: str) -> str:
    """Extract readable text from oEmbed HTML blockquote."""
    p_match = re.search(r"<p[^>]*>(.*?)</p>", html, re.DOTALL)
    if not p_match:
        return ""

    text = p_match.group(1)
    text = re.sub(r"<br\s*/?>", "\n", text)

    def _resolve_link(m: re.Match[str]) -> str:
        href_match = re.search(r'href="([^"]*)"', m.group(0))
        link_text = m.group(1)
        if "pic.twitter.com" in link_text:
            return "[image]"
        if href_match and "t.co/" in link_text:
            return href_match.group(1)
        return link_text

    text = re.sub(r"<a[^>]*>(.*?)</a>", _resolve_link, text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&mdash;", "—")
    return text.strip()


def fetch_tweet_oembed(client: httpx.Client, url: str) -> TweetData:
    """Fetch tweet text via the free oEmbed endpoint."""
    tweet = TweetData(url=url)

    try:
        resp = client.get(OEMBED_URL, params={"url": url, "omit_script": "true"})

        if resp.status_code == 404:
            tweet.fetch_error = "Tweet not found or private"
            return tweet

        if resp.status_code == 403:
            tweet.fetch_error = "Tweet is protected"
            return tweet

        if resp.status_code == 429:
            time.sleep(5)
            resp = client.get(OEMBED_URL, params={"url": url, "omit_script": "true"})
            if resp.status_code != 200:
                tweet.fetch_error = f"Rate limited (HTTP {resp.status_code})"
                return tweet

        if resp.status_code != 200:
            tweet.fetch_error = f"HTTP {resp.status_code}"
            return tweet

        data = resp.json()
        tweet.author_name = data.get("author_name", "")
        tweet.author_username = data.get("author_url", "").rstrip("/").split("/")[-1]

        html: str = data.get("html", "")
        tweet.text = _extract_text_from_oembed_html(html)

        # If tweet is just link(s), resolve them for context
        if tweet.text and _is_only_links(tweet.text):
            resolved_parts: list[str] = []
            for link in re.findall(r"https?://\S+", tweet.text):
                resolved = _resolve_tco_link(client, link)
                if "/article/" in resolved:
                    resolved_parts.append(f"[X Article] {resolved}")
                elif "x.com" in resolved or "twitter.com" in resolved:
                    resolved_parts.append(f"[Linked tweet/post] {resolved}")
                else:
                    resolved_parts.append(f"[External link] {resolved}")
            tweet.text = "\n".join(resolved_parts)

        if not tweet.text:
            tweet.fetch_error = "Empty tweet text"

    except httpx.TimeoutException:
        tweet.fetch_error = "Request timed out"
    except Exception as e:
        tweet.fetch_error = str(e)

    return tweet


# ---------------------------------------------------------------------------
# Ollama triage
# ---------------------------------------------------------------------------


def triage_tweet(tweet: TweetData, model: str) -> TriageResult:
    """Send a tweet to Ollama for categorization."""
    result = TriageResult(tweet=tweet)

    if tweet.fetch_error:
        result.category = "Fetch Error"
        result.summary = tweet.fetch_error
        return result

    user_msg = f"@{tweet.author_username} ({tweet.author_name})\n\n{tweet.text}"

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            options={"temperature": 0.1},
            format="json",
        )

        content = response["message"]["content"].strip()
        parsed = json.loads(content)
        result.category = parsed.get("category", "Other")
        result.high_priority = bool(parsed.get("high_priority", False))
        result.summary = parsed.get("summary", "")

    except (json.JSONDecodeError, KeyError) as e:
        result.llm_error = f"Parse error: {e}"
        result.category = "Parse Error"
    except Exception as e:
        result.llm_error = str(e)
        result.category = "LLM Error"

    return result


# ---------------------------------------------------------------------------
# Rich TUI
# ---------------------------------------------------------------------------


def build_display(state: TriageState, progress: Progress) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )

    # Header
    header_text = Text("  X Bookmarks Triage (oEmbed)", style="bold cyan")
    header_text.append(f"  [{state.processed}/{state.total}]", style="dim")
    layout["header"].update(Panel(header_text, style="cyan"))

    # Body
    layout["body"].split_row(
        Layout(name="left", ratio=2),
        Layout(name="right", ratio=3),
    )

    # Left — current tweet
    current_parts: list[Text] = []
    if state.current_tweet:
        current_parts.append(Text(f"@{state.current_author}", style="bold yellow"))
        current_parts.append(Text(""))
        display_text = state.current_tweet[:500]
        if len(state.current_tweet) > 500:
            display_text += "..."
        current_parts.append(Text(display_text, style="white"))
    else:
        current_parts.append(Text(state.current_phase, style="dim"))

    layout["left"].update(
        Panel(
            Group(*current_parts), title="Currently Processing", border_style="yellow"
        )
    )

    # Right — stats + recent
    right_layout = Layout()
    right_layout.split_column(
        Layout(name="stats", size=12),
        Layout(name="recent"),
    )

    stats_table = Table(show_header=False, box=None, padding=(0, 2))
    stats_table.add_column("Label", style="dim")
    stats_table.add_column("Value", style="bold")
    stats_table.add_row("Phase", state.current_phase)
    stats_table.add_row("Processed", f"{state.processed}/{state.total}")
    stats_table.add_row("Fetched OK", f"[green]{state.fetched}[/]")
    stats_table.add_row("Fetch Errors", f"[red]{state.fetch_errors}[/]")
    stats_table.add_row("High Priority", f"[bold red]{state.high_priority_count}[/]")
    stats_table.add_row("", "")

    sorted_cats = sorted(
        state.category_counts.items(), key=lambda x: x[1], reverse=True
    )[:4]
    for cat, count in sorted_cats:
        stats_table.add_row(cat, str(count))

    right_layout["stats"].update(
        Panel(stats_table, title="Stats", border_style="green")
    )

    recent_table = Table(show_header=True, expand=True)
    recent_table.add_column("Pri", width=3, justify="center")
    recent_table.add_column("Author", width=16, no_wrap=True)
    recent_table.add_column("Category", width=24, no_wrap=True)
    recent_table.add_column("Summary", ratio=1)

    for r in state.results[-8:]:
        pri = "[bold red]★[/]" if r.high_priority else " "
        author = f"@{r.tweet.author_username}" if r.tweet.author_username else "unknown"
        summary = (r.summary[:60] + "...") if len(r.summary) > 60 else r.summary
        recent_table.add_row(pri, author, r.category, summary)

    right_layout["recent"].update(
        Panel(recent_table, title="Recent Results", border_style="blue")
    )
    layout["right"].update(right_layout)

    # Footer
    layout["footer"].update(Panel(progress, style="dim"))

    return layout


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Triage X bookmarks with oEmbed API + Ollama"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_PATH, help="Output JSON path"
    )
    parser.add_argument(
        "--bookmarks", type=Path, default=BOOKMARKS_PATH, help="Bookmarks JSON path"
    )
    args = parser.parse_args()

    console = Console()

    # Load bookmarks
    bookmarks: list[str] = json.loads(args.bookmarks.read_text())
    console.print(f"Loaded [bold]{len(bookmarks)}[/] bookmarks")
    console.print("[dim]Using oEmbed API (free, no auth required)[/]")
    console.print()

    state = TriageState(total=len(bookmarks))

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        expand=True,
    )
    task_id = progress.add_task("Triaging bookmarks", total=len(bookmarks))

    with Live(
        build_display(state, progress), console=console, refresh_per_second=4
    ) as live:
        with httpx.Client(timeout=15.0) as client:
            for i, url in enumerate(bookmarks):
                # Phase 1: Fetch via Syndication API
                state.current_phase = f"Fetching ({i + 1}/{len(bookmarks)})"
                state.current_author = "..."
                state.current_tweet = ""
                live.update(build_display(state, progress))

                tweet = fetch_tweet_oembed(client, url)

                if tweet.fetch_error:
                    state.fetch_errors += 1
                else:
                    state.fetched += 1

                state.current_author = tweet.author_username or "unknown"
                state.current_tweet = tweet.text or f"[{tweet.fetch_error}]"

                # Phase 2: Triage with Ollama
                state.current_phase = f"Triaging ({i + 1}/{len(bookmarks)})"
                live.update(build_display(state, progress))

                result = triage_tweet(tweet, args.model)
                state.results.append(result)
                state.processed += 1

                if result.high_priority:
                    state.high_priority_count += 1

                state.category_counts[result.category] = (
                    state.category_counts.get(result.category, 0) + 1
                )

                progress.update(task_id, advance=1)
                live.update(build_display(state, progress))

                # Throttle requests
                time.sleep(FETCH_DELAY)

    # Save results
    output_data = {
        "metadata": {
            "total_bookmarks": len(bookmarks),
            "total_triaged": state.processed,
            "fetched_ok": state.fetched,
            "fetch_errors": state.fetch_errors,
            "high_priority_count": state.high_priority_count,
            "model": args.model,
            "categories": state.category_counts,
        },
        "high_priority": [
            {
                "url": r.tweet.url,
                "author": f"@{r.tweet.author_username}",
                "category": r.category,
                "summary": r.summary,
            }
            for r in state.results
            if r.high_priority
        ],
        "all_results": [
            {
                "url": r.tweet.url,
                "author": f"@{r.tweet.author_username}",
                "category": r.category,
                "high_priority": r.high_priority,
                "summary": r.summary,
                "text": r.tweet.text[:300],
                "error": r.llm_error or r.tweet.fetch_error,
            }
            for r in state.results
        ],
    }

    args.output.write_text(json.dumps(output_data, indent=2))

    # Final summary
    console.print()
    console.print(
        Panel.fit(
            f"[bold green]Done![/] Triaged [bold]{state.processed}[/] bookmarks.\n"
            f"Fetched [green]{state.fetched}[/] OK, [red]{state.fetch_errors}[/] errors.\n"
            f"[bold red]★ {state.high_priority_count}[/] flagged as high priority.\n"
            f"Results saved to [bold]{args.output}[/]",
            title="Triage Complete",
            border_style="green",
        )
    )

    if state.high_priority_count > 0:
        console.print()
        console.print("[bold red]★ High Priority Bookmarks:[/]")
        console.print()
        for r in state.results:
            if r.high_priority:
                console.print(f"  [bold]@{r.tweet.author_username}[/] — {r.summary}")
                console.print(f"  [dim]{r.tweet.url}[/]")
                console.print()


if __name__ == "__main__":
    main()
