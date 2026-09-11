"""
Diagnostic: compare pypdf vs PyMuPDF (fitz) text extraction on a specific
problem PDF, to check whether PyMuPDF fixes the chaotic spacing/kerning
issue before we decide whether to switch extraction libraries corpus-wide.

Usage:
    pip install pymupdf
    python test_pymupdf_extraction.py --pdf ./ctgov_protocols/NCT02051608.pdf
"""

import argparse
import fitz  # PyMuPDF


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--page", type=int, default=1, help="1-indexed page to sample (default: page 1)")
    args = parser.parse_args()

    doc = fitz.open(args.pdf)
    page = doc[args.page - 1]  # fitz is 0-indexed
    text = page.get_text()

    print(f"--- PyMuPDF extraction of page {args.page} ---\n")
    print(text[:2000])
    print("\n--- end sample ---")

    # quick check: does "Gantenerumab" or "endpoint" appear cleanly?
    # (case-insensitive, since body prose uses normal sentence case while
    # only the stylized title page uses all-caps)
    full_text = "\n".join(p.get_text() for p in doc)
    full_text_lower = full_text.lower()
    for term in ["gantenerumab", "endpoint", "with", "mild"]:
        found = term in full_text_lower
        print(f"'{term}' found as a clean substring (case-insensitive): {found}")

    print("\n--- context around first 'endpoint' occurrence in body text ---")
    idx = full_text_lower.find("endpoint")
    if idx >= 0:
        print(full_text[max(0, idx - 150) : idx + 150])
    else:
        print("Not found anywhere in the document.")
        # Check for character-level corruption: search for partial fragments
        # and print surrounding context wherever "Primary" or "Secondary"
        # appears near "Objective" -- endpoints are almost always discussed
        # right after objectives in protocol structure, so this should land
        # near the real endpoints section even if the word itself is corrupted.
        print("\n--- searching for 'Primary' near 'Objective' as a proxy location ---")
        obj_idx = full_text_lower.find("primary objective")
        if obj_idx >= 0:
            print(full_text[obj_idx : obj_idx + 600])
        else:
            print("'primary objective' also not found -- possible broader encoding issue.")


if __name__ == "__main__":
    main()
