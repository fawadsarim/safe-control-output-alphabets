# Minimum Output Alphabets for Safe Control Under Perception Completion Deadlines

Research code and manuscript for a joint IEEE Control Systems Letters (L-CSS) and ACC 2027
submission. **Status: submitted.**

[Manuscript source](paper/main.tex) | [Paper figure](paper/figures/information_completion.pdf) |
[Reproduce the results](#reproduce-the-results) | [Project layout](#project-layout)

## The problem in plain language

A controller must act while a perception result is still being computed. That result describes
the state when it was captured, not necessarily the state when the result arrives.

**How many distinct output labels are needed to keep the system safe and allow recovery, if the
result can arrive at any allowed time up to a guaranteed deadline?**

Here, an *output alphabet* is simply the finite set of labels a perception job can return. Each
label identifies a group of possible captured states. Grouping states under one label uses fewer
labels, but leaves the controller with uncertainty. The question is which groupings still
permit safe waiting and recovery at every possible completion time.

## What the paper establishes

| Question | Result |
| --- | --- |
| Can a given labeling support safe control while the result is pending? | An exact recursion tracks the possible states associated with each label. |
| What is the smallest feasible output alphabet? | A recovery-conflict hypergraph characterizes the minimum, jointly optimized with the pending controls. |
| Is checking every pair of states enough? | No. Larger groups can conflict even when every pair is compatible; pairwise tests can underestimate the required alphabet by an unbounded amount. |
| Does the labeling depend on completion age? | A fixed labeling can require arbitrarily more labels than one allowed to depend on completion age. |

The paper also establishes complexity and approximation results and a family with a precise
deadline-dependent increase in the required alphabet. See the [manuscript](paper/main.tex) for
the statements, assumptions, and proofs.

<details>
<summary>Formal contributions and computational limits</summary>

- Exact fixed-partition synthesis over propagated captured-label beliefs.
- An exact weak-hypergraph-coloring characterization of the minimum static output alphabet,
  minimized over pending controls, and a realization theorem for every finite conflict clutter.
- A recovery-word Set Cover construction with a polynomial-time harmonic-factor approximation
  for fixed recovery and completion horizons.
- An unbounded complete-ternary gap over pairwise tests, a sharp age-product bound, an unbounded
  static-versus-age-dependent encoder gap, and an exact deadline staircase.
- NP-completeness and inherited logarithmic inapproximability with one-step recovery at every
  fixed deadline.

For fixed horizons `H,d`, the constructive cover uses at most `R^(d+1)` candidates per pending
sequence, where `R=sum(r^h, h=0,...,H)` and `r` is the recovery alphabet size. Its runtime is
polynomial in the explicit plant table for each fixed `H,d`, not uniformly in variable horizons.
Exact partition and coloring search remains exponential.

</details>

## Scope and evidence

The model uses a finite nondeterministic plant, a static sound labeling of the captured state,
no new plant observations while a job is pending, guaranteed completion by a deadline, and
bounded-horizon, label-conditioned open-loop recovery after completion. These assumptions define the problem;
the algorithms do not establish that a physical perception system satisfies them.

Three differently structured exact solver routes are checked on **972 bounded-exhaustive
queries** and **600 seeded five-uncertain-state nondeterministic queries**. The tests also check
the paper's constructions and recovery witnesses. These are mathematical and computational
checks, not statistical evidence of physical safety. The enumeration oracle is separately
implemented, not independently authored.

There is no hardware, energy, worst-case execution-time, neural-network, flight, or empirical
safety claim. The research artifact runs on a standard PC; no GPU or specialized device is
required.

## Reproduce the results

The reference environment is **Windows PowerShell with CPython 3.12.10**. Dependencies are pinned
in [requirements-repro.txt](requirements-repro.txt). The instructions below use that environment;
they do not claim a separately validated Linux or macOS workflow.

### 1. Set up the environment

```powershell
git clone https://github.com/fawadsarim/safe-control-output-alphabets.git
cd safe-control-output-alphabets
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-repro.txt
.venv\Scripts\python.exe -m pip install --no-build-isolation --no-deps -e .
```

### 2. Run the checks and reproduce the figure

This runs lint, the test suite, the frozen finite experiment, numerical-claim checks, and figure
generation without requiring TeX:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/reproduce.ps1 -SkipPaper
```

### 3. Include the manuscript build

Install MiKTeX, or another TeX distribution exposing `pdflatex` and `bibtex`, then run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/reproduce.ps1
```

This repeats the computational checks and builds `output/pdf/lcss_acc_2027.pdf` from the frozen
manuscript source. The PDF checks include citation links, page count, and fonts in both the text
and embedded figures. Type 3 and unembedded fonts are rejected.

The full run must include the following success markers. The hash values below are placeholders
for the actual frozen hashes printed by the verifier:

```text
ARTIFACT_CLAIMS_PASS information_sha256=<frozen hash> figure_sha256=<frozen hash>
PAPER_DRAFT_GATE_PASS
PAPER_BUILD_PASS output=output/pdf/lcss_acc_2027.pdf
FULL_REPRODUCTION_PASS
```

`-SkipPaper` omits the paper checks and paper-build markers. To include the stricter
submission-source checks, use `-SubmissionReady` instead of `-SkipPaper`. That mode also checks
required contact data and AI-text disclosures/citations. Passing these automated checks is not
an editorial decision or a guarantee of acceptance.

<details>
<summary>Output files and reproducibility details</summary>

| Output | Location |
| --- | --- |
| Frozen experiment results | `artifacts/derived/information_completion_frontier.json` |
| Vector figure used in the paper | `paper/figures/information_completion.pdf` |
| Rebuilt manuscript PDF | `output/pdf/lcss_acc_2027.pdf` |

Generated JSON and the submission PDF are ignored so a checkout cannot accidentally pass using
stale outputs. The vector figure is tracked because it is part of the manuscript; the reproduction
command regenerates it before every build. The paper build fixes `SOURCE_DATE_EPOCH` to stabilize
PDF metadata; byte-identical PDF builds also depend on matching TeX tools, fonts, and build conditions.
The official unmodified `ieeeconf.cls` is downloaded
when absent and accepted only if its SHA-256 matches the value pinned in `scripts/build_paper.ps1`.

</details>

## Project layout

```text
paper/                              manuscript, bibliography, vector figure
src/safe_compute/                   exact finite solvers and witness constructions
tests/                              unit, property, and separate-oracle checks
experiments/pc/                     frozen finite experiment and figure generation
configs/                            frozen experiment parameters and seeds
scripts/reproduce.ps1               single end-to-end entry point
scripts/verify_artifacts.py         exact hashes and paper-number concordance
```

Useful entry points: [exact solvers](src/safe_compute/information_liveness.py),
[enumeration oracle](src/safe_compute/oracle.py), [recovery-cover construction](src/safe_compute/recovery_cover.py),
and [experiment configuration](configs/pc_information_completion.toml).

## Rebuild the manuscript

To run the manuscript-build entry point without rerunning the test suite:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build_paper.ps1 -SubmissionReady
```

This regenerates the numerical figure and builds `output/pdf/lcss_acc_2027.pdf` with source,
citation-link, font, and page-limit checks. The submitted manuscript is frozen; this command is
provided for reproduction, not to revise its scientific content.

## Author and license

Fawad Sarim ([ORCID: 0009-0002-8997-9288](https://orcid.org/0009-0002-8997-9288)).
See [LICENSE](LICENSE) for the repository's MIT license terms.

## Assistance provenance

OpenAI Codex assisted code implementation and testing, manuscript drafting and editing, and proof
development/inspection. The manuscript retains its acknowledgment and affected-section citations.
The author remains responsible for all scientific content and submission declarations.
