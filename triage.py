"""
X Bookmarks Triage — categorize and flag high-priority bookmarks using Ollama.

Uses the official X API v2 (https://docs.x.com/x-api/posts/lookup) for
fetching post data. Pay-per-use credits or Basic/Pro plan required.

Endpoints used:
  - GET /2/tweets       (batch lookup, up to 100 IDs per request)
  - GET /2/tweets/:id   (single post lookup, fallback)

Auth: Bearer Token (set as X_BEARER_TOKEN env var or in .env)
Docs: https://docs.x.com/x-api/posts/get-posts-by-ids

Requires:
  - X API v2 Bearer Token
  - Ollama running locally with a model pulled (default: llama3.2)

Usage:
  uv run triage.py
  uv run triage.py --model qwen3:14b
  uv run triage.py --output results.json
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import ollama
from dotenv import load_dotenv
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

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BOOKMARKS_PATH = Path(__file__).parent / "bookmarks.json"
DEFAULT_MODEL = "llama3.2"
OUTPUT_PATH = Path(__file__).parent / "triage_results.json"

# X API v2 — batch lookup supports up to 100 IDs per request
# https://docs.x.com/x-api/posts/get-posts-by-ids
X_API_BASE = "https://api.x.com/2"
X_API_BATCH_SIZE = 100
X_API_RATE_DELAY = 1.0  # seconds between batches

# Fields to request (official docs field names)
# https://docs.x.com/x-api/posts/get-post-by-id
TWEET_FIELDS = ",".join(
    [
        "author_id",
        "created_at",
        "entities",
        "note_tweet",  # full text for posts > 280 chars
        "public_metrics",  # like_count, retweet_count, reply_count, etc.
        "referenced_tweets",  # quoted/replied-to posts
        "article",  # X Articles (long-form content)
    ]
)

USER_FIELDS = ",".join(
    [
        "name",
        "username",
        "verified_type",
    ]
)

EXPANSIONS = ",".join(
    [
        "author_id",
        "referenced_tweets.id",
        "referenced_tweets.id.author_id",
    ]
)

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
    id: str
    url: str
    text: str = ""
    author_name: str = ""
    author_username: str = ""
    likes: int = 0
    retweets: int = 0
    replies: int = 0
    impressions: int = 0
    bookmarks: int = 0
    quotes: int = 0
    created_at: str = ""
    has_article: bool = False
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
    fetched_ok: int = 0
    fetch_errors: int = 0
    high_priority_count: int = 0
    current_tweet: str = ""
    current_author: str = ""
    current_phase: str = "Initializing..."
    results: list[TriageResult] = field(default_factory=list)
    category_counts: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tweet ID extraction
# ---------------------------------------------------------------------------


def extract_tweet_id(url: str) -> str | None:
    """Extract tweet/status ID (1-19 digits) from an X/Twitter URL."""
    match = re.search(r"/status/(\d{1,19})", url)
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# X API v2 — batch post lookup
# https://docs.x.com/x-api/posts/get-posts-by-ids
# ---------------------------------------------------------------------------


class RateLimitError(Exception):
    def __init__(self, retry_after: int):
        self.retry_after = retry_after
        super().__init__(f"Rate limited, retry after {retry_after}s")


def _expand_urls_in_text(text: str, entities: dict) -> str:
    """Replace t.co shortened URLs with their expanded versions using entity data."""
    for url_entity in entities.get("urls", []):
        short_url = url_entity.get("url", "")
        expanded = url_entity.get("expanded_url", short_url)
        if short_url and expanded:
            if "/article/" in expanded:
                text = text.replace(short_url, f"[X Article: {expanded}]")
            else:
                text = text.replace(short_url, expanded)
    return text


def fetch_tweets_batch(
    client: httpx.Client,
    tweet_ids: list[str],
    bearer_token: str,
) -> dict[str, TweetData]:
    """Fetch up to 100 posts via GET /2/tweets.

    Docs: https://docs.x.com/x-api/posts/get-posts-by-ids
    Auth: Bearer Token
    """
    results: dict[str, TweetData] = {}

    # GET https://api.x.com/2/tweets?ids=...&tweet.fields=...&user.fields=...&expansions=...
    resp = client.get(
        f"{X_API_BASE}/tweets",
        params={
            "ids": ",".join(tweet_ids),
            "tweet.fields": TWEET_FIELDS,
            "user.fields": USER_FIELDS,
            "expansions": EXPANSIONS,
        },
        headers={"Authorization": f"Bearer {bearer_token}"},
    )

    # Handle rate limiting (429)
    if resp.status_code == 429:
        retry_after = int(resp.headers.get("retry-after", "60"))
        raise RateLimitError(retry_after)

    if resp.status_code != 200:
        for tid in tweet_ids:
            results[tid] = TweetData(
                id=tid, url="", fetch_error=f"HTTP {resp.status_code}"
            )
        return results

    data = resp.json()

    # Build author lookup from includes.users
    authors: dict[str, dict[str, str]] = {}
    for user in data.get("includes", {}).get("users", []):
        authors[user["id"]] = {
            "name": user.get("name", ""),
            "username": user.get("username", ""),
        }

    # Build referenced tweet lookup from includes.tweets (for quoted posts)
    ref_tweets: dict[str, dict] = {}
    for rt in data.get("includes", {}).get("tweets", []):
        ref_tweets[rt["id"]] = rt

    # Process each post in the response
    for post in data.get("data", []):
        tid = post["id"]

        # note_tweet contains full text for posts > 280 chars
        note = post.get("note_tweet")
        if note and isinstance(note, dict):
            text = note.get("text", post.get("text", ""))
            # note_tweet has its own entities for URL expansion
            note_entities = note.get("entities", {})
            text = _expand_urls_in_text(text, note_entities)
        else:
            text = post.get("text", "")
            entities = post.get("entities", {})
            text = _expand_urls_in_text(text, entities)

        # Public metrics
        metrics = post.get("public_metrics", {})

        # Author info from expanded includes
        author = authors.get(post.get("author_id", ""), {})

        # Check for article content
        has_article = post.get("article") is not None

        # Append quoted tweet text for context
        for ref in post.get("referenced_tweets", []):
            if ref.get("type") == "quoted":
                quoted = ref_tweets.get(ref.get("id", ""))
                if quoted:
                    qt_author = authors.get(quoted.get("author_id", ""), {})
                    qt_text = quoted.get("text", "")
                    qt_entities = quoted.get("entities", {})
                    qt_text = _expand_urls_in_text(qt_text, qt_entities)
                    if qt_text:
                        qt_handle = qt_author.get("username", "?")
                        text += f"\n\n[Quoted @{qt_handle}]: {qt_text}"

        results[tid] = TweetData(
            id=tid,
            url="",
            text=text,
            author_name=author.get("name", ""),
            author_username=author.get("username", ""),
            likes=metrics.get("like_count", 0),
            retweets=metrics.get("retweet_count", 0),
            replies=metrics.get("reply_count", 0),
            impressions=metrics.get("impression_count", 0),
            bookmarks=metrics.get("bookmark_count", 0),
            quotes=metrics.get("quote_count", 0),
            created_at=post.get("created_at", ""),
            has_article=has_article,
        )

    # Handle errors for posts that weren't returned (deleted, private, etc.)
    errors_by_id: dict[str, str] = {}
    for err in data.get("errors", []):
        resource_id = err.get("resource_id") or err.get("value", "")
        errors_by_id[resource_id] = err.get("detail", err.get("title", "Unknown error"))

    for tid in tweet_ids:
        if tid not in results:
            results[tid] = TweetData(
                id=tid,
                url="",
                fetch_error=errors_by_id.get(tid, "Post not found or inaccessible"),
            )

    return results


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

    # Build a rich context string for the model
    parts: list[str] = []

    # Author line with metrics
    metrics_parts: list[str] = []
    if tweet.likes:
        metrics_parts.append(f"{tweet.likes:,} likes")
    if tweet.retweets:
        metrics_parts.append(f"{tweet.retweets:,} retweets")
    if tweet.bookmarks:
        metrics_parts.append(f"{tweet.bookmarks:,} bookmarks")
    if tweet.quotes:
        metrics_parts.append(f"{tweet.quotes:,} quotes")

    author_line = f"@{tweet.author_username} ({tweet.author_name})"
    if metrics_parts:
        author_line += f" | {', '.join(metrics_parts)}"
    parts.append(author_line)

    if tweet.has_article:
        parts.append("[This post contains an X Article (long-form content)]")

    parts.append("")
    parts.append(tweet.text)

    user_msg = "\n".join(parts)

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
    header_text = Text("  X Bookmarks Triage (API v2)", style="bold cyan")
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
            Group(*current_parts),
            title="Currently Processing",
            border_style="yellow",
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
    stats_table.add_row("Fetched OK", f"[green]{state.fetched_ok}[/]")
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
        author = (
            f"@{r.tweet.author_username}"
            if r.tweet.author_username
            else r.tweet.id[:12]
        )
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
        description="Triage X bookmarks with X API v2 + Ollama"
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
        "--bookmarks",
        type=Path,
        default=BOOKMARKS_PATH,
        help="Bookmarks JSON path",
    )
    args = parser.parse_args()

    console = Console()

    # Load bearer token
    bearer_token = os.environ.get("X_BEARER_TOKEN", "")
    if not bearer_token:
        console.print(
            "[bold red]Error:[/] X_BEARER_TOKEN not set. "
            "Set it in .env or as an environment variable."
        )
        console.print("Get one at https://developer.x.com/en/portal/dashboard")
        sys.exit(1)

    # Load bookmarks
    bookmarks: list[str] = json.loads(args.bookmarks.read_text())
    console.print(f"Loaded [bold]{len(bookmarks)}[/] bookmarks")
    console.print("[dim]Using X API v2 (Bearer Token auth)[/]")

    # Extract tweet IDs
    url_to_id: dict[str, str] = {}
    skipped: list[str] = []
    for url in bookmarks:
        tid = extract_tweet_id(url)
        if tid:
            url_to_id[url] = tid
        else:
            skipped.append(url)

    if skipped:
        console.print(f"[yellow]Skipped {len(skipped)} URLs (no tweet ID found)[/]")

    tweet_ids = list(url_to_id.values())
    urls_by_id = {v: k for k, v in url_to_id.items()}

    total_batches = (len(tweet_ids) + X_API_BATCH_SIZE - 1) // X_API_BATCH_SIZE
    console.print(
        f"Fetching [bold]{len(tweet_ids)}[/] posts in {total_batches} batch(es)..."
    )

    # Phase 1: Fetch all posts in batches
    all_tweets: dict[str, TweetData] = {}
    with httpx.Client(timeout=30.0) as client:
        for i in range(0, len(tweet_ids), X_API_BATCH_SIZE):
            batch = tweet_ids[i : i + X_API_BATCH_SIZE]
            batch_num = i // X_API_BATCH_SIZE + 1
            console.print(
                f"  Batch {batch_num}/{total_batches} ({len(batch)} posts)..."
            )

            try:
                fetched = fetch_tweets_batch(client, batch, bearer_token)
                for tid, tweet_data in fetched.items():
                    tweet_data.url = urls_by_id.get(tid, "")
                    all_tweets[tid] = tweet_data
            except RateLimitError as e:
                console.print(f"  [yellow]Rate limited. Waiting {e.retry_after}s...[/]")
                time.sleep(e.retry_after)
                # Retry the same batch
                fetched = fetch_tweets_batch(client, batch, bearer_token)
                for tid, tweet_data in fetched.items():
                    tweet_data.url = urls_by_id.get(tid, "")
                    all_tweets[tid] = tweet_data

            if i + X_API_BATCH_SIZE < len(tweet_ids):
                time.sleep(X_API_RATE_DELAY)

    fetched_ok = sum(1 for t in all_tweets.values() if not t.fetch_error)
    fetch_errors = sum(1 for t in all_tweets.values() if t.fetch_error)
    console.print(
        f"Fetched [bold green]{fetched_ok}[/] posts, [bold red]{fetch_errors}[/] errors"
    )
    console.print()

    # Phase 2: Triage with Ollama + live display
    state = TriageState(
        total=len(all_tweets), fetched_ok=fetched_ok, fetch_errors=fetch_errors
    )
    ordered_ids = [url_to_id[url] for url in bookmarks if url in url_to_id]

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        expand=True,
    )
    task_id = progress.add_task("Triaging bookmarks", total=len(ordered_ids))

    with Live(
        build_display(state, progress), console=console, refresh_per_second=4
    ) as live:
        for tid in ordered_ids:
            tweet = all_tweets.get(tid)
            if not tweet:
                continue

            state.current_author = tweet.author_username or tid[:12]
            state.current_tweet = tweet.text or "(no text)"
            state.current_phase = f"Triaging @{state.current_author}..."
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

    # Save results
    output_data = {
        "metadata": {
            "total_bookmarks": len(bookmarks),
            "total_triaged": state.processed,
            "fetched_ok": fetched_ok,
            "fetch_errors": fetch_errors,
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
                "likes": r.tweet.likes,
                "retweets": r.tweet.retweets,
                "bookmarks": r.tweet.bookmarks,
                "impressions": r.tweet.impressions,
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
                "likes": r.tweet.likes,
                "retweets": r.tweet.retweets,
                "bookmarks": r.tweet.bookmarks,
                "quotes": r.tweet.quotes,
                "impressions": r.tweet.impressions,
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
            f"Fetched [green]{fetched_ok}[/] OK, [red]{fetch_errors}[/] errors.\n"
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
