"""
Extract and clean text from downloaded ClinicalTrials.gov protocol PDFs.

Usage:
    python extract_and_clean.py --in_dir ./ctgov_protocols --out_dir ./ctgov_protocols_clean

What it does:
  1. Extracts text page-by-page from every PDF in --in_dir (keeps page numbers).
  2. Strips repeated boilerplate (footers, headers, version stamps) using a
     FREQUENCY-based approach: any short line that repeats across most pages
     of a document is almost certainly boilerplate, regardless of what sponsor-
     specific wording it uses. This generalizes across differently-formatted
     protocols instead of hardcoding one sponsor's footer text.
  3. Detects section headers using a flexible regex that catches multiple
     numbering styles (e.g. "4.0 SUBJECT SELECTION" and "3.1. Selection
     criteria"), while filtering out Table-of-Contents lines (dotted leaders
     or trailing page numbers).
  4. Writes one cleaned .txt file per protocol, with inline page markers
     (<<<PAGE n>>>) and detected section headers flagged, so the next step
     (chunking) can split on real section boundaries.
  5. Prints a per-document summary of what boilerplate was stripped and what
     headers were detected, so you can sanity-check before moving on.
"""

import argparse
import os
import re
from collections import Counter
from pypdf import PdfReader

# Matches section headers like "4.0 SUBJECT SELECTION" or "3.1. Selection criteria"
# or "4.2 Exclusion criteria" -- numeric prefix (with optional sub-levels and
# trailing period), followed by a short title-ish line.
SECTION_HEADER_RE = re.compile(
    r"^(?P<num>\d{1,2}(\.\d{1,2}){0,3}\.?)\s+(?P<title>[A-Za-z][A-Za-z0-9 \-/,'&()]{2,70})$"
)

# Lines that look like headers but are actually Table-of-Contents entries
# (dotted leaders or a trailing page number).
TOC_LIKE_RE = re.compile(r"\.{3,}|\.\s*\d{1,3}\s*$")


def extract_pages(pdf_path: str):
    """Return a list of (page_number, raw_text) tuples, 1-indexed."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append((i, text))
    return pages


def find_boilerplate_lines(pages, min_repeat_frac: float = 0.4, max_line_len: int = 120):
    """
    Identify lines that repeat across many pages -- these are almost always
    footers, headers, or version stamps rather than real content.

    A line qualifies as boilerplate if it:
      - is short (<= max_line_len chars, since real sentences run longer)
      - appears on at least `min_repeat_frac` of the document's pages
    """
    total_pages = len(pages)
    if total_pages < 3:
        return set()  # too few pages to detect a repeating pattern reliably

    line_counts = Counter()
    for _, text in pages:
        # dedupe within a single page so a line repeated twice on one page
        # doesn't inflate its cross-page count
        seen_this_page = set()
        for line in text.splitlines():
            line = line.strip()
            if line and len(line) <= max_line_len:
                seen_this_page.add(line)
        line_counts.update(seen_this_page)

    threshold = max(2, int(total_pages * min_repeat_frac))
    boilerplate = {line for line, count in line_counts.items() if count >= threshold}
    return boilerplate


def clean_and_annotate(pages, boilerplate: set):
    """
    Strip boilerplate lines and emit cleaned text with page markers and
    detected section headers flagged inline for the next (chunking) step.
    """
    out_lines = []
    detected_headers = []

    for page_num, text in pages:
        out_lines.append(f"<<<PAGE {page_num}>>>")
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped in boilerplate:
                continue

            match = SECTION_HEADER_RE.match(stripped)
            if match and not TOC_LIKE_RE.search(stripped):
                out_lines.append(f"<<<SECTION {stripped}>>>")
                detected_headers.append((page_num, stripped))
            out_lines.append(stripped)

    return "\n".join(out_lines), detected_headers


def process_document(pdf_path: str, out_dir: str):
    nct_id = os.path.splitext(os.path.basename(pdf_path))[0]
    pages = extract_pages(pdf_path)
    boilerplate = find_boilerplate_lines(pages)
    cleaned_text, headers = clean_and_annotate(pages, boilerplate)

    out_path = os.path.join(out_dir, f"{nct_id}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    return {
        "nct_id": nct_id,
        "pages": len(pages),
        "boilerplate_lines_removed": len(boilerplate),
        "boilerplate_sample": list(boilerplate)[:3],
        "headers_detected": len(headers),
        "header_sample": [h[1] for h in headers[:5]],
    }


def main():
    parser = argparse.ArgumentParser(description="Extract and clean ClinicalTrials.gov protocol PDFs.")
    parser.add_argument("--in_dir", default="./ctgov_protocols", help="Folder of downloaded protocol PDFs")
    parser.add_argument("--out_dir", default="./ctgov_protocols_clean", help="Where to write cleaned .txt files")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pdf_files = [f for f in os.listdir(args.in_dir) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"No PDFs found in {args.in_dir}")
        return

    print(f"Processing {len(pdf_files)} protocol PDFs...\n")
    summaries = []
    for fname in sorted(pdf_files):
        path = os.path.join(args.in_dir, fname)
        try:
            summary = process_document(path, args.out_dir)
            summaries.append(summary)
            print(
                f"[{summary['nct_id']}] {summary['pages']} pages | "
                f"{summary['boilerplate_lines_removed']} boilerplate lines removed | "
                f"{summary['headers_detected']} section headers detected"
            )
            if summary["header_sample"]:
                print(f"    sample headers: {summary['header_sample']}")
        except Exception as e:
            print(f"[{fname}] FAILED: {e}")

    total_headers = sum(s["headers_detected"] for s in summaries)
    zero_header_docs = [s["nct_id"] for s in summaries if s["headers_detected"] == 0]

    print(f"\nDone. Cleaned text written to {args.out_dir}/")
    print(f"Total section headers detected across corpus: {total_headers}")
    if zero_header_docs:
        print(
            f"\nWARNING: {len(zero_header_docs)} document(s) had ZERO section headers detected: "
            f"{zero_header_docs}\n"
            "These likely use a header format the regex doesn't catch yet (e.g. bold text with no "
            "numbering, or numbering embedded in a table). Worth opening these .txt files to check "
            "before moving to the chunking step."
        )


if __name__ == "__main__":
    main()
