# Week 2 RAG Eval: 15 Questions

Corpus: 24 Alzheimer's/MCI/dementia clinical trial protocols from ClinicalTrials.gov
(NCT02051608 excluded -- documented extraction failure, see report)

Run each with:
    python app.py --query "<question>"

For each answer, fill in the scoring columns below. Faithfulness = every
claim in the answer is actually supported by the cited chunk (check by
opening the chunk text, not just trusting the citation looks plausible).

---

## Category A: Single-trial factual (tests basic retrieval + citation)

1. **What is the primary endpoint of NCT03531710?**
   Sanity-check question -- already confirmed working. Include as your
   baseline "happy path" example in the report.

2. **What are the key inclusion criteria for the IVIG trial in amnestic mild cognitive impairment (NCT01300728)?**
   Tests retrieval from a long, numbered eligibility list (35 exclusion
   criteria in this doc) -- check the app doesn't just grab criterion #1
   and stop, but represents the list reasonably.

3. **What dose of donepezil (Aricept) is used in NCT01972204, and how is it titrated?**
   This document is a machine-translated-from-Japanese protocol with
   awkward phrasing (e.g. "Four)" instead of "4)"). Tests whether retrieval
   and generation hold up on non-native-English source text.

4. **What is the target sample size for NCT01972204, and what statistical assumption was used to calculate it?**
   Tests retrieval of a specific number buried in a methods section
   (Table 3 in this doc has multiple candidate percentages -- check the
   answer picks the actual target, not a random row from the power table).

5. **Who is the sponsor of NCT03531710, and what is the investigational drug?**
   Simple factual lookup, different section (Identification) than the
   endpoint questions -- tests whether retrieval finds sponsor/drug info
   which usually sits far from endpoint/eligibility content.

## Category B: Cross-document comparison (tests multi-hop retrieval)

6. **Compare the primary endpoints of NCT03531710 (UB-311) and NCT05189106 (baricitinib). Are they measuring the same kind of outcome?**
   Requires retrieving from two different trials in one query and holding
   both sets of citations straight without merging or confusing them.

7. **Which trials in this corpus use MMSE (Mini-Mental State Exam) as part of their eligibility or outcome assessment?**
   Open-ended cross-document question -- there's no single "right" chunk,
   so this tests whether hybrid retrieval surfaces multiple relevant trials
   rather than just the single best BM25/dense match.

8. **How do the exclusion criteria differ between the IVIG trial (NCT01300728) and the caregiver intervention study (NCT04603482)?**
   Deliberately pairs a heavily-regulated drug trial against a behavioral/
   observational study -- their eligibility criteria structures are very
   different (35 numbered exclusions vs. a handful of practical ones),
   good test of whether the app handles structural mismatch gracefully.

## Category C: Deliberately unanswerable (tests the refusal path)

9. **What is the primary endpoint of NCT02051608 (Gantenerumab)?**
   This document was excluded due to font-encoding corruption. The app
   should refuse -- this is your extraction-failure finding in action.
   Confirm the refusal message doesn't hallucinate an answer from a
   *different* Gantenerumab-adjacent trial instead.

10. **What is the recommended treatment for a patient who develops a rash during the NCT03531710 trial?**
    Plausible-sounding clinical question, but likely not a level of detail
    this protocol's retrieved chunks specifically cover. Tests whether the
    app refuses on a *specific* unanswerable detail rather than confidently
    generalizing from adjacent adverse-event language.

11. **What is the FDA's official position on donepezil dosing for severe Alzheimer's disease?**
    Outside-corpus question phrased to sound like it could be answered from
    protocol text (donepezil trial IS in the corpus) but asks for something
    the protocols don't contain (a regulatory position, not a trial's
    internal dosing schedule). Tests whether the app distinguishes "topic is
    in corpus" from "this specific claim is supported."

12. **What is the total enrollment target across all 24 trials in this corpus combined?**
    No single chunk answers this -- it requires aggregation the RAG app
    isn't designed to do. Should refuse or clearly flag that it can't
    aggregate across retrieved chunks, rather than inventing a number.

## Category D: Structural / edge-case stress tests

13. **Summarize the study design of NCT06210035.**
    This is the document with zero detected section headers (no numbered
    ToC), chunked via plain paragraph-splitting instead of section-aware
    chunking. Tests whether retrieval still works reasonably on a
    structurally "messy" document.

14. **What amendments were made to the IVIG trial protocol (NCT01300728) and why?**
    This document has 9 tracked amendments with rationale sections -- tests
    whether the chunker's section-awareness correctly separates amendment
    history from the main protocol body, and whether retrieval finds the
    *right* amendment if you ask a more specific follow-up.

15. **What is the definition of a Serious Adverse Event (SAE) used in NCT01972204, and how does it compare to the SAE definition in NCT03531710?**
    Combines a definitional-lookup question (tests precision on a specific
    clinical definition) with a cross-document comparison (Category B) --
    a good "hard mode" closer question that stresses multiple pipeline
    stages at once.

---

## Scoring template

For each question, record:

| # | Faithful? (Y/N) | Citations accurate? (Y/N) | Refused when it should? (Y/N/A) | Notes on failure mode |
|---|---|---|---|---|
| 1 |  |  |  |  |
| 2 |  |  |  |  |
| ... |  |  |  |  |  |

**Faithful?** -- open each cited chunk directly and verify the claim is actually
supported by that text, not just plausible-sounding.

**Citations accurate?** -- correct NCT ID and page number, not a hallucinated
or mismatched one.

**Refused when it should?** -- for Category C questions, did it correctly say
"I could not find this" rather than confidently answering from adjacent/wrong
content? (N/A for Categories A, B, D unless it refuses when it shouldn't.)

Aggregate faithfulness % across all 15 = your headline eval metric for the
Week 2 deliverable.
