"""
Chunk cleaned ClinicalTrials.gov protocol text (from extract_and_clean_v2.py)
into overlapping ~800-token chunks for embedding.

Usage:
    python chunk_documents.py --in_dir ./ctgov_protocols_clean --out_file ./chunks/chunks.jsonl

Strategy:
  - Split on <<<SECTION ...>>> boundaries where they exist, so a chunk never
    straddles two unrelated sections (e.g. Eligibility Criteria and
    Statistical Methods).
  - Within a section (or the whole doc, for section-less documents like
    NCT06210035), slide a fixed ~800-token window with ~100-token overlap.
    This is a hard cap that applies regardless of how good section detection
    was -- it's the safety net that makes the pipeline tolerant of imperfect
    header detection from the previous step.
  - Track token position -> page number per line so that a chunk deep inside
    a long section still cites the correct page, not just the section's
    starting page.

Output: one JSON object per line (JSONL), each with:
    nct_id, section, page, chunk_index, text, token_count
"""

import argparse
import json
import os
import re
import bisect
import tiktoken

PAGE_MARKER_RE = re.compile(r"^<<<PAGE (\d+)>>>$")
SECTION_MARKER_RE = re.compile(r"^<<<SECTION (.+)>>>$")

ENC = tiktoken.get_encoding("cl100k_base")


def parse_cleaned_file(path: str):
    """
    Parse a cleaned .txt file (with <<<PAGE n>>> and <<<SECTION title>>>
    markers) into a list of (page_num, section_name, line_text) tuples,
    one per real content line. Skips the duplicate content line that
    immediately follows a section marker (the header text itself), since
    it's redundant with the section metadata.
    """
    current_page = None
    current_section = "Preamble"
    just_started_section = False
    result = []

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue

            page_match = PAGE_MARKER_RE.match(line)
            if page_match:
                current_page = int(page_match.group(1))
                continue

            section_match = SECTION_MARKER_RE.match(line)
            if section_match:
                current_section = section_match.group(1).strip()
                just_started_section = True
                continue

            if just_started_section and line.strip() == current_section:
                # this is the duplicate header text right after the marker
                just_started_section = False
                continue
            just_started_section = False

            result.append((current_page, current_section, line.strip()))

    return result


def group_into_sections(lines_with_meta):
    """Group consecutive lines sharing the same section into blocks."""
    blocks = []
    current_section = None
    current_page = None
    current_lines = []

    for page, section, text in lines_with_meta:
        if section != current_section:
            if current_lines:
                blocks.append((current_section, current_page, current_lines))
            current_section = section
            current_page = page
            current_lines = []
        current_lines.append((page, text))

    if current_lines:
        blocks.append((current_section, current_page, current_lines))

    return blocks


def chunk_block(section_name: str, lines, max_tokens: int, overlap_tokens: int):
    """
    Slide a token window over a section's lines, tracking token offset -> page
    so each chunk can cite the correct page even deep inside a long section.
    """
    full_token_ids = []
    line_page_starts = []  # list of (cum_token_offset, page)
    cum = 0

    for page, text in lines:
        token_ids = ENC.encode(text + " ")
        line_page_starts.append((cum, page))
        full_token_ids.extend(token_ids)
        cum += len(token_ids)

    total = len(full_token_ids)
    if total == 0:
        return []

    chunks = []
    start = 0
    offsets = [p[0] for p in line_page_starts]

    while start < total:
        end = min(start + max_tokens, total)
        chunk_token_ids = full_token_ids[start:end]
        chunk_text = ENC.decode(chunk_token_ids)

        idx = bisect.bisect_right(offsets, start) - 1
        idx = max(0, idx)
        page = line_page_starts[idx][1]

        chunks.append(
            {
                "section": section_name,
                "page": page,
                "text": chunk_text.strip(),
                "token_count": len(chunk_token_ids),
            }
        )

        if end == total:
            break
        start = end - overlap_tokens  # slide with overlap

    return chunks


def chunk_document(path: str, max_tokens: int, overlap_tokens: int):
    nct_id = os.path.splitext(os.path.basename(path))[0]
    lines_with_meta = parse_cleaned_file(path)
    blocks = group_into_sections(lines_with_meta)

    all_chunks = []
    for section_name, _, lines in blocks:
        block_chunks = chunk_block(section_name, lines, max_tokens, overlap_tokens)
        all_chunks.extend(block_chunks)

    for i, c in enumerate(all_chunks):
        c["nct_id"] = nct_id
        c["chunk_index"] = i

    return all_chunks


def main():
    parser = argparse.ArgumentParser(description="Chunk cleaned protocol text into overlapping token windows.")
    parser.add_argument("--in_dir", default="./ctgov_protocols_clean")
    parser.add_argument("--out_file", default="./chunks/chunks.jsonl")
    parser.add_argument("--max_tokens", type=int, default=800)
    parser.add_argument("--overlap_tokens", type=int, default=100)
    parser.add_argument(
        "--exclude",
        default="",
        help="Comma-separated NCT IDs to skip (e.g. known-bad PDFs with corrupted extraction). "
        "Example: --exclude NCT02051608,NCT01234567",
    )
    args = parser.parse_args()

    excluded_ids = {x.strip() for x in args.exclude.split(",") if x.strip()}
    if excluded_ids:
        print(f"Excluding {len(excluded_ids)} document(s): {sorted(excluded_ids)}\n")

    os.makedirs(os.path.dirname(args.out_file) or ".", exist_ok=True)
    txt_files = [f for f in os.listdir(args.in_dir) if f.lower().endswith(".txt")]
    txt_files = [f for f in txt_files if os.path.splitext(f)[0] not in excluded_ids]

    if not txt_files:
        print(f"No cleaned .txt files found in {args.in_dir}")
        return

    print(f"Chunking {len(txt_files)} documents (max_tokens={args.max_tokens}, overlap={args.overlap_tokens})...\n")

    all_chunks = []
    per_doc_counts = {}
    for fname in sorted(txt_files):
        path = os.path.join(args.in_dir, fname)
        doc_chunks = chunk_document(path, args.max_tokens, args.overlap_tokens)
        all_chunks.extend(doc_chunks)
        nct_id = os.path.splitext(fname)[0]
        per_doc_counts[nct_id] = len(doc_chunks)
        avg_tokens = sum(c["token_count"] for c in doc_chunks) / len(doc_chunks) if doc_chunks else 0
        print(f"[{nct_id}] {len(doc_chunks)} chunks | avg {avg_tokens:.0f} tokens/chunk")

    with open(args.out_file, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    total_tokens = sum(c["token_count"] for c in all_chunks)
    print(f"\nDone. {len(all_chunks)} total chunks written to {args.out_file}")
    print(f"Total tokens across corpus: {total_tokens:,}")
    print(f"Avg chunk size: {total_tokens / len(all_chunks):.0f} tokens" if all_chunks else "")

    empty_docs = [nct for nct, count in per_doc_counts.items() if count == 0]
    if empty_docs:
        print(f"\nWARNING: {len(empty_docs)} document(s) produced ZERO chunks: {empty_docs}")


if __name__ == "__main__":
    main()
