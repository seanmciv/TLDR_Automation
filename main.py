#!/usr/bin/env python3
"""
TLDR AI Bot: Scrapes last 7 archive entries, summarizes top 5 via LLM, emails via Resend.
"""

import logging

from dotenv import load_dotenv
from pathlib import Path
# Load .env from tldr-bot dir, then parent (TL;DR Agent)
load_dotenv()
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
import os
import re
from typing import Optional

import requests
from bs4 import BeautifulSoup
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ARCHIVES_URL = "https://tldr.tech/ai/archives"
BASE_URL = "https://tldr.tech"
DAILY_LINK_PATTERN = re.compile(r"/ai/20\d{2}-\d{2}-\d{2}")


def get_archive_links(limit: int = 7) -> list[str]:
    """Fetch archives page and return the first `limit` daily edition URLs."""
    resp = requests.get(ARCHIVES_URL, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    links = []
    seen = set()
    for a in soup.find_all("a", href=True):
        href = a.get("href", "").strip()
        if DAILY_LINK_PATTERN.search(href):
            full_url = href if href.startswith("http") else (BASE_URL + href)
            if full_url not in seen:
                seen.add(full_url)
                links.append(full_url)
        if len(links) >= limit:
            break

    return links[:limit]


def extract_stories_from_daily(soup: BeautifulSoup, date_str: str) -> list[dict]:
    """Extract story blocks from a daily page. Skip sponsors and job ads."""
    stories = []
    # Look for h3 headings (### in the content) - story titles
    for h3 in soup.find_all(["h3", "h4"]):
        title = h3.get_text(strip=True)
        if not title:
            continue
        # Skip sponsor and job blocks
        lower = title.lower()
        if "sponsor" in lower or "hiring" in lower or "tldr is hiring" in lower:
            continue

        # Get body: next siblings until next heading
        body_parts = []
        for sib in h3.next_siblings:
            if sib.name in ("h2", "h3", "h4"):
                break
            if sib.name == "p":
                body_parts.append(sib.get_text(strip=True))
            elif hasattr(sib, "get_text") and sib.name in ("div", "section"):
                t = sib.get_text(strip=True)
                if t and len(t) > 20:
                    body_parts.append(t)

        body = " ".join(body_parts) if body_parts else ""
        if title and (body or len(title) > 15):
            stories.append({"date": date_str, "title": title, "body": body})

    return stories


def fetch_daily_content(url: str) -> list[dict]:
    """Fetch a daily TLDR page and return extracted stories."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Extract date from URL (e.g. /ai/2026-01-23 -> 2026-01-23)
    date_match = re.search(r"/ai/(20\d{2}-\d{2}-\d{2})", url)
    date_str = date_match.group(1) if date_match else "unknown"

    return extract_stories_from_daily(soup, date_str)


def build_context_for_llm(all_stories: list[dict]) -> str:
    """Build a single text blob for the LLM from all stories."""
    parts = []
    for s in all_stories:
        block = f"## {s['date']}\n### {s['title']}\n{s['body']}"
        parts.append(block)
    return "\n\n---\n\n".join(parts)


def _load_prompts() -> tuple[str, str]:
    """Load system and user prompt templates from prompts/ folder."""
    prompts_dir = Path(__file__).resolve().parent / "prompts"
    system_path = prompts_dir / "tldr_report_system.txt"
    user_path = prompts_dir / "tldr_report_user.txt"
    system = system_path.read_text(encoding="utf-8").strip()
    user_template = user_path.read_text(encoding="utf-8")
    return system, user_template


def summarize_with_llm(context: str) -> Optional[str]:
    """Use OpenAI to pick top 5 most impactful items for a dev team."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set. Cannot summarize.")
        return None

    client = OpenAI(api_key=api_key)
    system_prompt, user_template = _load_prompts()
    full_prompt = user_template + context

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt},
            ],
            max_tokens=1500,
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.exception("LLM call failed: %s", e)
        return None


def markdown_to_html(md: str) -> str:
    """Convert simple Markdown to minimal HTML for email."""
    lines = md.split("\n")
    out = []
    for line in lines:
        s = line.strip()
        if not s:
            out.append("<br/>")
            continue
        # Bold
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"__(.+?)__", r"<strong>\1</strong>", s)
        # Headers
        if s.startswith("### "):
            out.append(f"<h3>{s[4:]}</h3>")
        elif s.startswith("## "):
            out.append(f"<h2>{s[3:]}</h2>")
        elif s.startswith("- "):
            out.append(f"<li>{s[2:]}</li>")
        else:
            out.append(f"<p>{s}</p>")
    html = "\n".join(out)
    # Wrap consecutive <li> in <ul>
    html = re.sub(r"(<li>.*?</li>\n?)+", lambda m: "<ul>\n" + m.group(0) + "</ul>", html)
    return "<html><head><meta charset='utf-8'></head><body>" + html + "</body></html>"


def send_email_summary(summary_md: str) -> bool:
    """Send the summary via Resend transactional email API."""
    import resend

    api_key = os.environ.get("RESEND_API_KEY")
    email_from = os.environ.get("EMAIL_FROM")
    email_to = os.environ.get("EMAIL_TO")

    if not all([api_key, email_from, email_to]):
        missing = []
        if not api_key:
            missing.append("RESEND_API_KEY")
        if not email_from:
            missing.append("EMAIL_FROM")
        if not email_to:
            missing.append("EMAIL_TO")
        logger.error("Missing env vars for email: %s. Add them to .env", ", ".join(missing))
        return False

    subject = "TLDR AI – Top 5 This Week"
    html_content = markdown_to_html(summary_md)
    text_content = summary_md.replace("**", "").replace("__", "")

    resend.api_key = api_key
    params = {
        "from": f"TLDR Bot <{email_from}>" if "@" in email_from and "<" not in email_from else email_from,
        "to": [email_to],
        "subject": subject,
        "html": html_content,
        "text": text_content,
    }

    try:
        resend.Emails.send(params)
        logger.info("Email sent successfully to %s via Resend", email_to)
        return True
    except Exception as e:
        logger.exception("Resend email failed: %s", e)
        return False


def save_log(summary: str, sent: bool, dry_run: bool) -> None:
    """Save the generated summary to logs/ with a datestamped filename."""
    from datetime import datetime, timezone
    logs_dir = Path(__file__).resolve().parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    status = "dry-run" if dry_run else ("sent" if sent else "failed")
    filename = logs_dir / f"{ts}_{status}.md"
    header = f"# TLDR AI Summary\n\nDate: {ts} UTC  \nStatus: {status}\n\n---\n\n"
    filename.write_text(header + summary, encoding="utf-8")
    logger.info("Log saved to %s", filename)


def main(dry_run: bool = False) -> None:
    logger.info("Fetching archive links...")
    links = get_archive_links(limit=7)
    logger.info("Found %d daily links", len(links))

    all_stories = []
    for url in links:
        try:
            stories = fetch_daily_content(url)
            all_stories.extend(stories)
            logger.info("Fetched %d stories from %s", len(stories), url)
        except Exception as e:
            logger.warning("Failed to fetch %s: %s", url, e)

    if not all_stories:
        logger.error("No stories extracted. Exiting.")
        return

    context = build_context_for_llm(all_stories)
    logger.info("Context length: %d chars", len(context))

    summary = summarize_with_llm(context)
    if not summary:
        logger.error("Summarization failed. Exiting.")
        return

    if dry_run:
        print("\n--- Generated Summary (dry run, email not sent) ---\n")
        print(summary)
        save_log(summary, sent=False, dry_run=True)
        logger.info("Dry run complete.")
        return

    logger.info("Summary generated. Sending email...")
    sent = send_email_summary(summary)
    save_log(summary, sent=sent, dry_run=False)
    if sent:
        logger.info("Done.")
    else:
        logger.error("Email delivery failed.")
        print("\n--- Generated Summary ---\n")
        print(summary)


if __name__ == "__main__":
    import sys
    main(dry_run="--dry-run" in sys.argv)
