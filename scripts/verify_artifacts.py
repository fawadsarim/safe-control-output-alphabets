"""Fail closed if regenerated artifacts do not match the frozen manuscript claims."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

INFORMATION_PATH = Path("artifacts/derived/information_completion_frontier.json")
FIGURE_PATH = Path("paper/figures/information_completion.pdf")

# Updated only after two independent-process regenerations agree byte for byte.
EXPECTED_INFORMATION_SHA256 = "a505b15e300bc820f5e13547f583723e040f214a366c0dd33c4e39f7b9775d23"
EXPECTED_FIGURE_SHA256 = "50ef0fdd15907b24a055e38e0985e9123df7797bdb8606c16c33fc464d3fff83"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    information = _load(INFORMATION_PATH)
    _require(_sha256(INFORMATION_PATH) == EXPECTED_INFORMATION_SHA256, "information SHA mismatch")
    _require(_sha256(FIGURE_PATH) == EXPECTED_FIGURE_SHA256, "figure SHA mismatch")
    _require(information["bounded_exhaustive"] == {"plants": 81, "queries": 972}, "bounded audit")
    _require(
        information["three_way_crosschecks"] == 600,
        "five-uncertain-state three-way cross-checks",
    )
    _require(
        [row["minimum_symbols"] for row in information["structured_frontier"]]
        == [1, 2, 3, 4],
        "staircase frontier",
    )
    _require(
        [row["minimum_symbols"] for row in information["higher_order_gap"]]
        == [2, 2, 3, 3, 4, 4],
        "complete-ternary frontier",
    )
    _require(
        all(row["pairwise_graph_estimate"] == 1 for row in information["higher_order_gap"]),
        "complete-ternary pairwise estimate",
    )
    _require(
        [row["static_symbols"] for row in information["static_age_gap"]]
        == [2, 3, 4, 5, 6, 7]
        and all(row["known_age_symbols"] == 2 for row in information["static_age_gap"]),
        "static-versus-known-age gap",
    )
    manuscript = Path("paper/main.tex").read_text(encoding="utf-8")
    compact = re.sub(r"\s+", " ", manuscript)
    for claim in (
        "$3^4=81$ transition tables, three nonempty launch beliefs, and four deadlines: "
        "all 972 queries agree",
        "a further 600 five-uncertain-state nondeterministic queries agree",
        "Tests also exhaust all 19 clutters on three vertices",
        "the cover dual and greedy certificate are checked on all 167 clutters over four "
        "vertices",
        "it is NP-hard to approximate finite $q^*(X,d)$ within $(1-\\alpha)\\ln n$",
    ):
        _require(claim in compact, f"manuscript claim drift: {claim}")

    print(
        "ARTIFACT_CLAIMS_PASS "
        f"information_sha256={_sha256(INFORMATION_PATH)} "
        f"figure_sha256={_sha256(FIGURE_PATH)}"
    )


if __name__ == "__main__":
    main()
