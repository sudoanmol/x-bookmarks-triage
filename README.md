# x-bookmarks-triage

Auto-triage your X/Twitter bookmarks using a local LLM. Categorizes each bookmark, flags high-priority posts (quality articles, actionable advice, research breakthroughs), and generates a structured report.

## How it works

1. Reads bookmark URLs from `bookmarks.json`
2. Fetches tweet content (text, author, metrics, quoted tweets, expanded URLs)
3. Sends each tweet to a local Ollama model for categorization
4. Displays a real-time TUI with progress, current tweet, and running stats
5. Outputs `triage_results.json` with categories, summaries, and priority flags

### Categories

Posts are classified into one of:

- Technical Article / Tutorial
- AI / ML Research & News
- Career & Life Advice
- Product Launch / Announcement
- Open Source / Developer Tool
- Design & UI/UX
- Business & Startup
- Humor / Meme
- Crypto / Web3
- Health & Wellness
- News / Current Events
- Other

### High priority criteria

A post is flagged as high priority if it contains genuinely valuable information: deep technical content, actionable advice, important announcements, or research worth revisiting. Memes, generic motivational content, and low-effort threads are not flagged.

## Scripts

There are three scripts, each using a different data source:

| Script                  | Data source       | Auth required | Data quality                                          |
| ----------------------- | ----------------- | ------------- | ----------------------------------------------------- |
| `triage_free.py`        | X Syndication API | None          | Full text, metrics, expanded URLs, quoted tweets      |
| `triage_free_oembed.py` | oEmbed API        | None          | Text only, no metrics                                 |
| `triage.py`             | X API v2          | Bearer Token  | Richest -- all metrics, article detection, note_tweet |

**Start with `triage_free.py`** -- it requires no API keys and returns rich data. Use `triage.py` if you have X API v2 access for the most complete results. `triage_free_oembed.py` exists as a fallback if the syndication API stops working.

## Setup

### Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- [Ollama](https://ollama.com/) running locally with a model pulled

### Install

```sh
# Clone and install dependencies
git clone <repo-url> && cd x-bookmarks
uv sync

# Install and start Ollama (macOS)
brew install ollama
brew services start ollama

# Pull a model
ollama pull llama3.2        # 3B, fast, ~4GB RAM
ollama pull qwen3:8b        # 8B, good balance, ~6GB RAM
ollama pull qwen3:14b       # 14B, best quality, ~10GB RAM
```

### Bookmarks file

Export your bookmarks from X using the included DevTools script:

1. Go to https://x.com/i/bookmarks in your browser
2. Open DevTools (Cmd+Option+J on Mac, Ctrl+Shift+J on Windows/Linux)
3. Paste the contents of `export_bookmarks.js` into the Console and press Enter
4. Wait while it auto-scrolls through all your bookmarks
5. A `bookmarks.json` file will download automatically when done
6. Move it to the project root

The file is a JSON array of post URLs:

```json
["https://x.com/user/status/1234567890", "https://x.com/user/status/0987654321"]
```

### X API v2 setup (optional, for `triage.py` only)

```sh
cp .env.example .env
# Edit .env with your Bearer Token from https://developer.x.com/en/portal/dashboard
```

## Usage

```sh
# Free, no auth required (recommended)
uv run triage_free.py

# With a specific model
uv run triage_free.py --model qwen3:14b

# Custom output path
uv run triage_free.py --output my_results.json

# Using X API v2 (requires Bearer Token in .env)
uv run triage.py --model qwen3:14b

# oEmbed fallback
uv run triage_free_oembed.py
```

## Output

Results are saved to `triage_results.json` with the following structure:

```json
{
  "metadata": {
    "total_bookmarks": 432,
    "total_triaged": 432,
    "high_priority_count": 47,
    "model": "qwen3:14b",
    "categories": {
      "AI / ML Research & News": 89,
      "Technical Article / Tutorial": 72,
      "Open Source / Developer Tool": 65
    }
  },
  "high_priority": [
    {
      "url": "https://x.com/user/status/...",
      "author": "@username",
      "category": "AI / ML Research & News",
      "summary": "Deep dive into transformer attention mechanisms with code examples.",
      "likes": 12500,
      "retweets": 3200
    }
  ],
  "all_results": [...]
}
```

## Model recommendations

Performance depends on available RAM (Apple Silicon unified memory or GPU VRAM):

| RAM   | Model           | Notes                                   |
| ----- | --------------- | --------------------------------------- |
| 8GB   | `llama3.2` (3B) | Fast, decent quality                    |
| 12GB  | `qwen3:8b`      | Good balance of speed and intelligence  |
| 16GB+ | `qwen3:14b`     | Best for structured output / JSON tasks |
| 32GB+ | `gemma3:27b`    | Near-frontier quality                   |

`qwen3:14b` is recommended for this task -- Qwen3 excels at structured JSON output which is exactly what the triage prompt requires.

## Limitations

- **X Articles**: Long-form X Articles are behind a JS-rendered wall. The syndication and oEmbed APIs return the article URL but not its full content. The X API v2 can detect articles via the `article` field but also cannot return the body text.
- **Deleted/private posts**: Posts that have been deleted or made private will be logged as fetch errors and skipped during triage.
- **Rate limits**: The syndication API has undocumented rate limits. The script includes automatic backoff and retry logic, but very large bookmark collections may need multiple runs.
