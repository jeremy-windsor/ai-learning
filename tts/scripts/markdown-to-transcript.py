#!/usr/bin/env python3
"""Convert AI learning markdown into spoken transcript text.

This is intentionally conservative: preserve lesson content, remove markdown
noise, make headings into spoken section breaks, and leave pronunciation to
apply-pronunciation.py.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def clean_inline(s: str) -> str:
    s = s.strip()
    # Drop HTML comments.
    s = re.sub(r"<!--.*?-->", "", s)
    # Markdown images -> spoken placeholder omitted.
    s = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", s)
    # Links -> keep label only unless URL is the label.
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", s)
    # Inline code -> bare text.
    s = re.sub(r"`([^`]+)`", r"\1", s)
    # Bold/italic markers.
    s = s.replace("**", "").replace("__", "")
    s = re.sub(r"(?<!\w)[*_](.*?)[*_](?!\w)", r"\1", s)
    # Emoji / dingbats common in docs; remove instead of making TTS weird.
    s = re.sub(r"[✅❌⚠️🔥🧠🗿👉📌📅📰🚀⭐💡🎯📚🔧🧪]+", "", s)
    # Collapse whitespace.
    s = re.sub(r"\s+", " ", s).strip()
    return s


def heading_text(level: int, title: str) -> str:
    title = clean_inline(title)
    if not title:
        return ""
    if level == 1:
        return title
    if level == 2:
        return f"Section: {title}."
    if level == 3:
        return f"Topic: {title}."
    return f"{title}."


def convert(path: Path) -> str:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    out: list[str] = []
    para: list[str] = []
    in_fence = False
    table_run = False

    def flush_para() -> None:
        nonlocal para
        if para:
            text = clean_inline(" ".join(para))
            if text:
                out.append(text)
            para = []

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()

        if stripped.startswith("```") or stripped.startswith("~~~"):
            flush_para()
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        if not stripped:
            flush_para()
            table_run = False
            continue

        # Skip markdown table separator and table rows. Tables are usually too
        # awkward for audio; the lesson prose already explains the point.
        if stripped.startswith("|") and stripped.endswith("|"):
            flush_para()
            table_run = True
            continue

        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            flush_para()
            table_run = False
            h = heading_text(len(m.group(1)), m.group(2))
            if h:
                out.append(h)
            continue

        # Horizontal rule.
        if re.fullmatch(r"[-*_]{3,}", stripped):
            flush_para()
            table_run = False
            continue

        # Blockquotes: keep content, remove quote marker.
        stripped = re.sub(r"^>\s?", "", stripped)

        # Bullets/tasks -> short spoken sentence.
        bullet = re.match(r"^[-*+]\s+(.*)$", stripped)
        numbered = re.match(r"^\d+[.)]\s+(.*)$", stripped)
        task = re.match(r"^- \[[ xX]\]\s+(.*)$", stripped)
        if task:
            flush_para()
            item = clean_inline(task.group(1))
            if item:
                out.append(item + ("." if not item.endswith(('.', '?', '!')) else ""))
            continue
        if bullet or numbered:
            flush_para()
            item = clean_inline((bullet or numbered).group(1))
            if item:
                out.append(item + ("." if not item.endswith(('.', '?', '!')) else ""))
            continue

        para.append(stripped)

    flush_para()

    # Final inline cleanup for TTS-hostile syntax that may be meaningful in
    # markdown but awful when spoken literally.
    for i, item in enumerate(out):
        item = item.replace("<|", "").replace("|>", "")
        item = item.replace("\\<", "less than ").replace("\\>", " greater than ")
        item = item.replace("<", " less than ").replace(">", " greater than ")
        item = item.replace("~", " approximately ")
        out[i] = re.sub(r"\s+", " ", item).strip()

    # Clean punctuation artifacts.
    text = "\n\n".join(x for x in out if x)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(convert(args.input), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
