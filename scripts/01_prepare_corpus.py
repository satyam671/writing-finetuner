"""
01_prepare_corpus.py

Clean your raw writing exports into plain text.
Handles Medium HTML exports, plain .txt, and .md files.

Usage:
    python scripts/01_prepare_corpus.py \
        --input_dir data/raw \
        --output_dir data/cleaned \
        --min_words 200
"""

import argparse
import os
import re
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup
import markdown
from tqdm import tqdm


def clean_html(filepath: Path) -> str:
    """Parse a Medium HTML export and extract clean article body text."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()

    soup = BeautifulSoup(raw, "lxml")

    # Medium exports wrap body in <article> or <section>
    article = soup.find("article") or soup.find("section") or soup.body

    if article is None:
        return ""

    # Drop elements that aren't prose
    for tag in article.find_all(["script", "style", "nav", "footer",
                                   "header", "figure", "figcaption",
                                   "button", "form", "aside"]):
        tag.decompose()

    # Extract paragraphs and headings in order
    blocks = []
    for tag in article.find_all(["h1", "h2", "h3", "h4", "p", "li", "blockquote", "pre"]):
        text = tag.get_text(separator=" ", strip=True)
        if text:
            blocks.append(text)

    return "\n\n".join(blocks)


def clean_markdown(filepath: Path) -> str:
    """Convert Markdown to plain text by stripping all markup."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()

    # Convert MD to HTML, then strip tags
    html = markdown.markdown(raw)
    soup = BeautifulSoup(html, "lxml")
    return soup.get_text(separator="\n\n", strip=True)


def clean_plaintext(filepath: Path) -> str:
    """Read and normalize a plain text file."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def normalize_text(text: str) -> str:
    """
    Normalize unicode, collapse whitespace, strip common cruft.
    Keeps sentence-level structure intact.
    """
    # Normalize unicode (NFKC handles smart quotes, ligatures, etc.)
    text = unicodedata.normalize("NFKC", text)

    # Strip zero-width and non-printing characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

    # Collapse multiple blank lines to one
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Normalize horizontal whitespace within lines
    text = re.sub(r"[ \t]+", " ", text)

    # Strip trailing whitespace per line
    lines = [line.rstrip() for line in text.splitlines()]
    text = "\n".join(lines)

    return text.strip()


def word_count(text: str) -> int:
    return len(text.split())


def process_file(filepath: Path) -> str:
    """Dispatch to the right cleaner based on file extension."""
    ext = filepath.suffix.lower()

    if ext == ".html" or ext == ".htm":
        text = clean_html(filepath)
    elif ext == ".md":
        text = clean_markdown(filepath)
    elif ext == ".txt":
        text = clean_plaintext(filepath)
    else:
        # Try plain text for anything else
        text = clean_plaintext(filepath)

    return normalize_text(text)


def main():
    parser = argparse.ArgumentParser(description="Prepare writing corpus for fine-tuning.")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing raw writing exports")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write cleaned plain text files")
    parser.add_argument("--min_words", type=int, default=200,
                        help="Minimum word count to keep a file (default: 200)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    supported = {".html", ".htm", ".md", ".txt"}
    files = [f for f in input_dir.rglob("*") if f.suffix.lower() in supported]

    if not files:
        print(f"No supported files found in {input_dir}. Expected .html, .md, or .txt")
        return

    print(f"Found {len(files)} files. Cleaning...")

    kept, skipped = 0, 0
    total_words = 0

    for filepath in tqdm(files, desc="Cleaning"):
        try:
            text = process_file(filepath)
        except Exception as e:
            print(f"  WARN: Failed to process {filepath.name}: {e}")
            skipped += 1
            continue

        wc = word_count(text)
        if wc < args.min_words:
            skipped += 1
            continue

        out_path = output_dir / (filepath.stem + ".txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)

        kept += 1
        total_words += wc

    print(f"\nDone.")
    print(f"  Files kept:    {kept}")
    print(f"  Files skipped: {skipped} (below {args.min_words} words or parse error)")
    print(f"  Total words:   {total_words:,}")
    print(f"  Output:        {output_dir}/")


if __name__ == "__main__":
    main()
