"""
Extract and clean text from downloaded ClinicalTrials.gov protocol PDFs (v2).

Key change from v1: instead of using a generic regex to guess which lines are
section headers (which produced many false positives -- eligibility list items
and addresses were being misread as headers), this version first extracts each
document's own Table of Contents and uses it as a whitelist of real section
titles. Only lines in the body that match a ToC entry are tagged as section
boundaries. This is far more precise because list items and addresses never
appear in a ToC.

Usage:
    python extract_and_clean_v2.py --in_dir ./ctgov_protocols --out_dir ./ctgov_protocols_clean
"""

import argparse
import os
import re
from collections import Counter
from pypdf import PdfReader

# --- De-kerning: some PDFs render headers with a space injected between every
# letter (e.g. "S T U D Y"). Detect lines where most tokens are single
# characters and collapse them back into words.
def dekern_line(line: str) -> str:
    tokens = line.split(" ")
    if len(tokens) < 4:
        return line
    single_char_frac = sum(1 for t in tokens if len(t) == 1) / len(tokens)
    if single_char_frac < 0.6:
        return line
    result, buffer = [], ""
    for t in tokens:
        if len(t) == 1 and t.isalnum():
            buffer += t
        else:
            if buffer:
                result.append(buffer)
                buffer = ""
            result.append(t)
    if buffer:
        result.append(buffer)
    return " ".join(result)


# --- ToC line patterns: "Title .......... 12" (dot leaders) or "Title    12"
# (space-separated, no dots), optionally preceded by a section number.
TOC_DOTLEADER_RE = re.compile(
    r"^(?P<num>\d{1,2}(\.\d{1,2}){0,3}\.?)?\s*(?P<title>[A-Za-z][A-Za-z0-9 ,\-/&()']{2,80}?)"
    r"\s*\.{3,}\s*(?P<page>\d{1,4})$"
)
TOC_SPACED_RE = re.compile(
    r"^(?P<num>\d{1,2}(\.\d{1,2}){0,3}\.?)\s+(?P<title>[A-Za-z][A-Za-z0-9 ,\-/&()']{2,80}?)"
    r"\s+(?P<page>\d{1,4})$"
)

# Weak trailing words that indicate a line is a truncated sentence, not a header
WEAK_TRAILING_WORDS = {
    "a", "an", "the", "of", "to", "in", "on", "or", "and", "with", "for",
    "by", "at", "as", "is", "be", "at least",
}


def extract_pages(pdf_path: str):
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append((i, text))
    return pages


def find_boilerplate_lines(pages, min_repeat_frac: float = 0.4, max_line_len: int = 120):
    total_pages = len(pages)
    if total_pages < 3:
        return set()
    line_counts = Counter()
    for _, text in pages:
        seen_this_page = set()
        for line in text.splitlines():
            line = dekern_line(line.strip())
            if line and len(line) <= max_line_len:
                seen_this_page.add(line)
        line_counts.update(seen_this_page)
    threshold = max(2, int(total_pages * min_repeat_frac))
    return {line for line, count in line_counts.items() if count >= threshold}


def extract_toc_entries(pages, toc_page_limit: int = 20):
    """
    Scan the first `toc_page_limit` pages for ToC-style lines and return a
    whitelist of (normalized_title_prefix) strings. We key on title text
    rather than section number since numbering styles are inconsistent
    across sponsors, but titles are unique and stable.
    """
    entries = []
    for page_num, text in pages[:toc_page_limit]:
        for line in text.splitlines():
            line = dekern_line(line.strip())
            if not line:
                continue
            m = TOC_DOTLEADER_RE.match(line) or TOC_SPACED_RE.match(line)
            if m:
                title = m.group("title").strip()
                if len(title.split()) >= 1:
                    entries.append(title)
    # Normalize for matching: lowercase, collapse whitespace
    normalized = {re.sub(r"\s+", " ", t).strip().lower() for t in entries}
    return normalized


def is_real_header(line: str, toc_titles: set) -> bool:
    """Check if a body line's title portion matches a known ToC entry."""
    m = re.match(
        r"^(?P<num>\d{1,2}(\.\d{1,2}){0,3}\.?)\s+(?P<title>[A-Za-z][A-Za-z0-9 ,\-/&()']{2,80})$",
        line,
    )
    if not m:
        return False
    title = re.sub(r"\s+", " ", m.group("title")).strip().lower()
    # exact match or the body title starts with a known ToC title (handles
    # minor truncation/wrapping differences)
    if title in toc_titles:
        return True
    return any(title.startswith(t) or t.startswith(title) for t in toc_titles if len(t) > 4)


def fallback_is_header(line: str) -> bool:
    """
    Used only when no ToC was found for a document. Tightened vs. v1:
    caps word count, rejects lines ending in a weak/truncated word.
    """
    m = re.match(
        r"^(?P<num>\d{1,2}(\.\d{1,2}){0,3}\.?)\s+(?P<title>[A-Za-z][A-Za-z0-9 ,\-/&()']{2,70})$",
        line,
    )
    if not m:
        return False
    title = m.group("title").strip()
    words = title.split()
    if len(words) > 8:
        return False
    if words[-1].lower().strip(".,") in WEAK_TRAILING_WORDS:
        return False
    return True


def clean_and_annotate(pages, boilerplate: set, toc_titles: set):
    out_lines = []
    detected_headers = []
    used_fallback = len(toc_titles) == 0

    for page_num, text in pages:
        out_lines.append(f"<<<PAGE {page_num}>>>")
        for raw_line in text.splitlines():
            line = dekern_line(raw_line.strip())
            if not line or line in boilerplate:
                continue

            is_header = is_real_header(line, toc_titles) if toc_titles else fallback_is_header(line)
            if is_header:
                out_lines.append(f"<<<SECTION {line}>>>")
                detected_headers.append((page_num, line))
            out_lines.append(line)

    return "\n".join(out_lines), detected_headers, used_fallback


def process_document(pdf_path: str, out_dir: str, min_toc_entries: int = 5):
    nct_id = os.path.splitext(os.path.basename(pdf_path))[0]
    pages = extract_pages(pdf_path)
    boilerplate = find_boilerplate_lines(pages)
    toc_titles = extract_toc_entries(pages)

    # If ToC extraction found too few entries to be a reliable whitelist,
    # treat it as if no ToC was found at all and use the fallback regex
    # instead -- a near-empty whitelist is worse than no whitelist, since it
    # silently suppresses real headers rather than just missing some.
    if 0 < len(toc_titles) < min_toc_entries:
        toc_titles = set()

    cleaned_text, headers, used_fallback = clean_and_annotate(pages, boilerplate, toc_titles)

    out_path = os.path.join(out_dir, f"{nct_id}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    return {
        "nct_id": nct_id,
        "pages": len(pages),
        "toc_entries_found": len(toc_titles),
        "boilerplate_lines_removed": len(boilerplate),
        "headers_detected": len(headers),
        "header_sample": [h[1] for h in headers[:5]],
        "used_fallback": used_fallback,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract and clean ClinicalTrials.gov protocol PDFs (v2).")
    parser.add_argument("--in_dir", default="./ctgov_protocols")
    parser.add_argument("--out_dir", default="./ctgov_protocols_clean")
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
            mode = "FALLBACK (no ToC found)" if summary["used_fallback"] else f"ToC ({summary['toc_entries_found']} entries)"
            print(
                f"[{summary['nct_id']}] {summary['pages']} pages | mode: {mode} | "
                f"{summary['headers_detected']} headers detected"
            )
            if summary["header_sample"]:
                print(f"    sample: {summary['header_sample']}")
        except Exception as e:
            print(f"[{fname}] FAILED: {e}")

    fallback_docs = [s["nct_id"] for s in summaries if s["used_fallback"]]
    zero_header_docs = [s["nct_id"] for s in summaries if s["headers_detected"] == 0]

    print(f"\nDone. Cleaned text written to {args.out_dir}/")
    if fallback_docs:
        print(
            f"\n{len(fallback_docs)} document(s) had no detectable Table of Contents, "
            f"so the less-reliable fallback regex was used: {fallback_docs}\n"
            "Worth opening these specifically to check header quality by eye."
        )
    if zero_header_docs:
        print(f"\n{len(zero_header_docs)} document(s) got ZERO headers detected: {zero_header_docs}")


if __name__ == "__main__":
    main()
