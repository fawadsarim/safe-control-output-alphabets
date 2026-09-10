# Minimum Output Alphabets for Safe Control Under Perception Completion Deadlines

Reproducible code and manuscript for the joint IEEE Control Systems Letters (L-CSS) and ACC 2027
submission.

The paper studies a finite nondeterministic plant when a perception job labels the state at
capture, the label may arrive at any age up to a guaranteed deadline, and recovery must remain
possible at every possible completion age. It provides:

- an exact fixed-partition recursion over propagated captured-label beliefs;
- an exact weak-hypergraph-coloring characterization of the minimum static output alphabet,
  jointly minimized over pending controls, plus a realization theorem for every finite conflict
  clutter and a recovery-word Set Cover construction with a polynomial-time harmonic-factor
  approximation for fixed recovery and completion horizons;
- an unbounded complete-ternary gap over all pairwise tests, a sharp age-product bound and
  unbounded static-versus-age-dependent encoder gap, NP-completeness and inherited logarithmic
  inapproximability with one-step recovery at every fixed deadline, and an exact deadline staircase; and
- three exact solver routes checked on a complete bounded class and seeded nondeterministic plants.

For fixed horizons `H,d`, the constructive cover uses at most `R^(d+1)` candidates per pending
sequence, where `R=sum(r^h, h=0,...,H)` and `r` is the recovery alphabet size. Its runtime is
polynomial in the explicit plant table for each fixed `H,d`, not uniformly in variable horizons.

The formal scope is deliberately narrow. There is no hardware, energy, WCET, neural-network,
flight, or empirical safety claim. The complete artifact runs on a standard PC with Python and a
TeX distribution; no specialized embedded device is required.

## Reproduce from a fresh clone

The frozen environment is Windows PowerShell with CPython 3.12.10. MiKTeX (or another TeX
distribution exposing `pdflatex` and `bibtex`) is needed for the paper build.

```powershell
git clone https://github.com/fawadsarim/safe-control-output-alphabets.git
cd safe-control-output-alphabets
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-repro.txt
.venv\Scripts\python.exe -m pip install --no-build-isolation --no-deps -e .
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/reproduce.ps1
```

The command runs lint, the complete test suite, the frozen finite experiment, fail-closed numerical
claim checks, figure generation, and the validated PDF build. The PDF gate checks fonts in embedded
figures as well as the page text, rejecting Type 3 and unembedded fonts. Its final lines must include:

```text
ARTIFACT_CLAIMS_PASS information_sha256=<frozen hash> figure_sha256=<frozen hash>
PAPER_DRAFT_GATE_PASS
PAPER_BUILD_PASS ...\output\pdf\lcss_acc_2027.pdf
FULL_REPRODUCTION_PASS
```

Generated JSON and the submission PDF are ignored so a checkout cannot accidentally pass using
stale outputs. The vector figure is tracked because it is part of the manuscript; the reproduction
command regenerates it before every build. The paper build fixes `SOURCE_DATE_EPOCH` to stabilize
PDF metadata; byte-identical PDF builds also depend on matching TeX tools, fonts, and build conditions.
The official unmodified `ieeeconf.cls` is downloaded
when absent and accepted only if its SHA-256 matches the value pinned in `scripts/build_paper.ps1`.

For a code-only run while TeX is being installed, append `-SkipPaper`. For the final submission
gate, run with `-SubmissionReady`. That stricter mode fails on either missing contact data or missing required
AI-text disclosures/citations. `FULL_REPRODUCTION_PASS` certifies reproduction, not venue compliance.

## Project layout

```text
paper/                              manuscript, bibliography, vector figure
src/safe_compute/                   exact finite solvers and witness constructions
tests/                              unit, property, and independent-oracle checks
experiments/pc/                     frozen finite experiment and figure generation
configs/                            immutable experiment parameters and seeds
scripts/reproduce.ps1               single end-to-end entry point
scripts/verify_artifacts.py         exact hashes and paper-number concordance
```

The scientific tests include examples and solver comparisons reported in the paper. They and the
build/consistency checks are part of reproduction; private review reports and one-off audit
programs are not included. The oracle is separately implemented, not independently authored.

## Rebuild the manuscript

To rebuild only the paper with its final source, citation-link, font, and page-limit checks:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build_paper.ps1 -SubmissionReady
```

The output is `output/pdf/lcss_acc_2027.pdf`. The manuscript source includes the supplied author
contact details; review those before publishing its source.

## Assistance provenance

OpenAI Codex assisted code implementation and testing, manuscript drafting and editing, and proof
development/inspection. The manuscript retains its acknowledgment and affected-section citations.
The author remains responsible for all scientific content and submission declarations.
