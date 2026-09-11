"""
Fetch clinical trial protocol PDFs from ClinicalTrials.gov for a given
therapeutic area / condition.

Usage:
    python fetch_ctgov_protocols.py --condition "non-small cell lung cancer" --target 25

What it does:
  1. Queries the ClinicalTrials.gov API v2 for studies matching a condition.
  2. Filters down to studies that have an uploaded protocol PDF.
  3. Downloads protocol PDFs (stopping once --target is reached).
  4. Saves a metadata CSV (nct_id, title, phase, filename, download status)
     so you know exactly what's in your corpus.

Docs: https://clinicaltrials.gov/data-api/api
"""

import argparse
import csv
import os
import time
import requests

API_URL = "https://clinicaltrials.gov/api/v2/studies"
CDN_BASE = "https://cdn.clinicaltrials.gov/large-docs"


def fetch_candidate_studies(condition: str, page_size: int = 100, max_pages: int = 10):
    """Page through the API collecting studies for a condition.

    Note: deliberately NOT restricting the `fields` param here. The API's
    field-name aliasing for nested pieces (like documentSection) is easy to
    get wrong and will silently return empty data instead of erroring, which
    is what caused the "0 protocols found" bug. Fetching full records costs
    a bit more bandwidth but guarantees nothing gets dropped.
    """
    studies = []
    params = {
        "query.cond": condition,
        "pageSize": page_size,
    }
    page_token = None
    for _ in range(max_pages):
        if page_token:
            params["pageToken"] = page_token
        resp = requests.get(API_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        studies.extend(data.get("studies", []))
        page_token = data.get("nextPageToken")
        if not page_token:
            break
        time.sleep(0.2)  # be polite to the API
    return studies


def debug_print_first_study(studies):
    """Print the raw documentSection of the first study for troubleshooting."""
    if not studies:
        print("[debug] No studies returned at all.")
        return
    import json
    s = studies[0]
    nct_id = s.get("protocolSection", {}).get("identificationModule", {}).get("nctId", "?")
    doc_section = s.get("documentSection", {})
    print(f"[debug] First study: {nct_id}")
    print(f"[debug] documentSection raw: {json.dumps(doc_section, indent=2)[:1000]}")


def studies_with_protocol(studies):
    """Filter to studies that have an uploaded protocol document (not just SAP or ICF)."""
    filtered = []
    for s in studies:
        protocol_section = s.get("protocolSection", {})
        ident = protocol_section.get("identificationModule", {})
        design = protocol_section.get("designModule", {})
        nct_id = ident.get("nctId")
        title = ident.get("briefTitle", "")
        phase = ", ".join(design.get("phases", [])) if design.get("phases") else "N/A"

        # documentSection is a TOP-LEVEL sibling of protocolSection in the
        # API response, not nested inside it. This was the bug causing 0
        # protocol matches — we were looking in the wrong place entirely.
        docs = (
            s.get("documentSection", {})
            .get("largeDocumentModule", {})
            .get("largeDocs", [])
        )
        # typeAbbrev "Prot" = Protocol, "Prot_SAP" = Protocol + Statistical Analysis Plan
        protocol_docs = [
            d for d in docs if d.get("typeAbbrev", "").startswith("Prot") and d.get("filename")
        ]
        if nct_id and protocol_docs:
            filtered.append(
                {
                    "nct_id": nct_id,
                    "title": title,
                    "phase": phase,
                    "doc": protocol_docs[0],  # take the first protocol doc if multiple
                }
            )
    return filtered


def build_download_url(nct_id: str, filename: str) -> str:
    # CDN layout: large-docs/{last 2 digits of NCT number}/{NCT_ID}/{filename}
    last_two = nct_id[-2:]
    return f"{CDN_BASE}/{last_two}/{nct_id}/{filename}"


def download_protocols(records, target: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    results = []
    downloaded = 0

    for rec in records:
        if downloaded >= target:
            break

        nct_id = rec["nct_id"]
        filename = rec["doc"]["filename"]
        url = build_download_url(nct_id, filename)
        local_path = os.path.join(out_dir, f"{nct_id}.pdf")

        status = "skipped"
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200 and resp.content[:4] == b"%PDF":
                with open(local_path, "wb") as f:
                    f.write(resp.content)
                status = "downloaded"
                downloaded += 1
            else:
                status = f"failed (HTTP {resp.status_code})"
        except requests.RequestException as e:
            status = f"failed ({e})"

        results.append(
            {
                "nct_id": nct_id,
                "title": rec["title"],
                "phase": rec["phase"],
                "filename": filename,
                "url": url,
                "status": status,
            }
        )
        time.sleep(0.2)  # be polite to the CDN
        print(f"[{downloaded}/{target}] {nct_id}: {status}")

    return results


def write_metadata_csv(results, out_dir: str):
    csv_path = os.path.join(out_dir, "corpus_metadata.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["nct_id", "title", "phase", "filename", "url", "status"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nMetadata written to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Fetch ClinicalTrials.gov protocol PDFs for a condition.")
    parser.add_argument("--condition", required=True, help='e.g. "non-small cell lung cancer"')
    parser.add_argument("--target", type=int, default=25, help="Number of protocols to download")
    parser.add_argument("--out_dir", default="./ctgov_protocols", help="Output directory")
    parser.add_argument("--debug", action="store_true", help="Print raw API response shape for troubleshooting")
    args = parser.parse_args()

    print(f"Querying ClinicalTrials.gov for condition: '{args.condition}'...")
    studies = fetch_candidate_studies(args.condition)
    print(f"Found {len(studies)} total studies matching condition.")

    if args.debug:
        debug_print_first_study(studies)

    candidates = studies_with_protocol(studies)
    print(f"Of those, {len(candidates)} have an uploaded protocol PDF.")

    if len(candidates) < args.target:
        print(
            f"\nWARNING: only {len(candidates)} candidates available, "
            f"fewer than your target of {args.target}.\n"
            "Consider broadening the condition term or adding a related condition."
        )

    results = download_protocols(candidates, args.target, args.out_dir)
    write_metadata_csv(results, args.out_dir)

    succeeded = sum(1 for r in results if r["status"] == "downloaded")
    print(f"\nDone. {succeeded}/{args.target} protocols downloaded to {args.out_dir}/")


if __name__ == "__main__":
    main()
