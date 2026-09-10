"""Static and PDF checks for the joint L-CSS/ACC manuscript."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

EXPECTED_CLASS = r"\documentclass[letterpaper,10pt,conference]{ieeeconf}"
MAX_ABSTRACT_WORDS = 200
MAX_PAGES = 6
US_LETTER_POINTS = (612.0, 792.0)


def _without_comments(source: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in source.splitlines())


def _abstract_word_count(source: str) -> int:
    match = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
        source,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError("missing abstract environment")
    text = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^]]*\])?", " ", match.group(1))
    text = re.sub(r"[{}$~_^]", " ", text)
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", text))


def _citation_keys(source: str) -> set[str]:
    groups = re.findall(r"\\cite(?:t|p)?\{([^}]+)\}", source)
    return {key.strip() for group in groups for key in group.split(",") if key.strip()}


def _bibliography_keys(source: str) -> set[str]:
    return set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", source, flags=re.IGNORECASE))


def _reference_urls(source: str) -> set[str]:
    # Release references use explicit, braced, plain-text URL fields. IEEEtran
    # renders these; its standard .bst does not render a bare DOI field.
    return set(re.findall(r"\burl\s*=\s*\{([^{}\s]+)\}", source, flags=re.IGNORECASE))


def validate_sources(tex_path: Path, bib_path: Path, *, submission_ready: bool) -> list[str]:
    errors: list[str] = []
    tex_raw = tex_path.read_text(encoding="utf-8")
    tex = _without_comments(tex_raw)
    bib = _without_comments(bib_path.read_text(encoding="utf-8"))

    if EXPECTED_CLASS not in tex:
        errors.append(f"document class must be exactly: {EXPECTED_CLASS}")

    try:
        abstract_words = _abstract_word_count(tex)
    except ValueError as exc:
        errors.append(str(exc))
    else:
        print(f"ABSTRACT_WORDS {abstract_words}/{MAX_ABSTRACT_WORDS}")
        if abstract_words > MAX_ABSTRACT_WORDS:
            errors.append(f"abstract has {abstract_words} words; maximum is {MAX_ABSTRACT_WORDS}")

    missing = sorted(_citation_keys(tex) - _bibliography_keys(bib))
    if missing:
        errors.append(f"missing bibliography keys: {', '.join(missing)}")
    all_keys = re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib, flags=re.IGNORECASE)
    duplicates = sorted(key for key, count in Counter(all_keys).items() if count > 1)
    if duplicates:
        errors.append(f"duplicate bibliography keys: {', '.join(duplicates)}")
    unused = sorted(_bibliography_keys(bib) - _citation_keys(tex))
    if unused:
        message = f"uncited bibliography keys: {', '.join(unused)}"
        if submission_ready:
            errors.append(message)
        else:
            print(f"PAPER_DRAFT_WARNING {message}")

    todo_markers = sorted(set(re.findall(r"TODO-[A-Z-]+", tex_raw)))
    if todo_markers:
        message = f"unresolved markers: {', '.join(todo_markers)}"
        if submission_ready:
            errors.append(message)
        else:
            print(f"PAPER_DRAFT_WARNING {message}")

    # Draft builds may intentionally omit disclosures, but that must never be
    # mistaken for submission compliance. The strict gate preserves the venue
    # requirement even when a citation-free draft is requested.
    policy_errors: list[str] = []
    acknowledgment = re.search(
        r"\\section\*\{Acknowledgment\}(.*?)(?:\\bibliographystyle|\\end\{document\})",
        tex,
        flags=re.DOTALL,
    )
    body = tex.split(r"\section*{Acknowledgment}", maxsplit=1)[0]
    if "openai2026codex" not in _citation_keys(body):
        policy_errors.append("AI-assistance citation is missing from the manuscript body")
    sections = re.split(r"\\section\{[^}]*\}", body)
    for index, section in enumerate(sections[1:], start=1):
        if "openai2026codex" not in _citation_keys(section):
            policy_errors.append(f"AI-assistance citation is missing from section {index}")
    # Project-specific provenance: assistance here included drafting, code, and
    # proof arguments. A grammar-only notice would understate this project's use;
    # these scope checks are not a universal rule about every IEEE manuscript.
    disclosure_fragments = (
        "OpenAI Codex",
        "manuscript drafting and editing throughout the paper",
        "code development",
        "drafting and inspecting proof arguments",
        "sole responsibility",
    )
    acknowledgment_text = (
        "" if acknowledgment is None else re.sub(r"\s+", " ", acknowledgment.group(1))
    )
    if any(fragment not in acknowledgment_text for fragment in disclosure_fragments):
        policy_errors.append("AI-assistance disclosure is missing from Acknowledgment")
    if submission_ready:
        errors.extend(policy_errors)
    else:
        for issue in policy_errors:
            print(f"PAPER_DRAFT_WARNING {issue}; L-CSS submission compliance is not met")
    return errors


def validate_log(log_path: Path) -> list[str]:
    if not log_path.exists():
        return [f"LaTeX log does not exist: {log_path}"]
    log = log_path.read_text(encoding="utf-8", errors="replace")
    errors: list[str] = []
    checks = {
        r"Overfull \\hbox": "overfull horizontal box",
        r"Overfull \\vbox": "overfull vertical box",
        r"LaTeX Error": "LaTeX error",
        r"Citation .* undefined": "undefined citation",
        r"Reference .* undefined": "undefined reference",
        r"There were undefined references": "undefined references",
    }
    for pattern, description in checks.items():
        if re.search(pattern, log, flags=re.IGNORECASE):
            errors.append(f"LaTeX log contains {description}")
    return errors


def validate_font_resources(resources, *, seen: set[int] | None = None) -> list[str]:
    """Inspect fonts in pages and nested figure Form XObjects, without external tools."""

    errors: list[str] = []
    if resources is None:
        return errors
    resources = resources.get_object()
    seen = set() if seen is None else seen
    if id(resources) in seen:
        return errors
    seen.add(id(resources))
    fonts = resources.get("/Font", {})
    fonts = fonts.get_object() if hasattr(fonts, "get_object") else fonts
    for reference in fonts.values():
        font = reference.get_object()
        name = font.get("/BaseFont", "unnamed")
        if font.get("/Subtype") == "/Type3":
            errors.append(f"PDF contains Type 3 font: {name}")
            continue
        faces = font.get("/DescendantFonts", [font])
        for face in faces:
            descriptor = face.get_object().get("/FontDescriptor")
            if descriptor is not None:
                descriptor = descriptor.get_object()
            if descriptor is None or not any(
                key in descriptor for key in ("/FontFile", "/FontFile2", "/FontFile3")
            ):
                errors.append(f"PDF font is not embedded: {name}")
    objects = resources.get("/XObject", {})
    objects = objects.get_object() if hasattr(objects, "get_object") else objects
    for reference in objects.values():
        obj = reference.get_object()
        if obj.get("/Subtype") == "/Form":
            errors.extend(validate_font_resources(obj.get("/Resources"), seen=seen))
    return errors


def validate_bibtex_log(log_path: Path) -> list[str]:
    if not log_path.exists():
        return [f"BibTeX log does not exist: {log_path}"]
    log = log_path.read_text(encoding="utf-8", errors="replace")
    if re.search(
        r"^Warning--|Repeated entry|I didn't find a database entry|"
        r"You're missing a field name|I was expecting a|error message",
        log,
        flags=re.MULTILINE | re.IGNORECASE,
    ):
        return ["BibTeX log contains a warning or error"]
    return []


def validate_reference_links(pdf_path: Path, bib_path: Path) -> list[str]:
    """Check actual PDF link targets, not merely text that looks like a URL.

    This is an offline structural check; it does not replace a live DOI audit.
    """
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    bib = _without_comments(bib_path.read_text(encoding="utf-8"))
    keys = _bibliography_keys(bib)
    expected_urls = _reference_urls(bib)
    errors: list[str] = []
    if len(expected_urls) != len(keys):
        errors.append("release bibliography must provide one distinct explicit URL per entry")
    actual_urls: set[str] = set()
    actual_citations: set[str] = set()
    for page in reader.pages:
        for reference in page.get("/Annots", []):
            annotation = reference.get_object()
            if isinstance(annotation.get("/Dest"), str):
                actual_citations.add(str(annotation["/Dest"]))
            action = annotation.get("/A")
            if action is not None:
                action = action.get_object()
                if action.get("/S") == "/URI":
                    actual_urls.add(str(action.get("/URI", "")))
                elif action.get("/S") == "/GoTo" and isinstance(action.get("/D"), str):
                    actual_citations.add(str(action["/D"]))
    for url in sorted(expected_urls - actual_urls):
        errors.append(f"PDF is missing reference hyperlink: {url}")
    destinations = reader.named_destinations
    for key in sorted(keys):
        if f"cite.{key}" not in destinations:
            errors.append(f"PDF is missing bibliography destination: {key}")
        if f"cite.{key}" not in actual_citations:
            errors.append(f"PDF is missing citation hyperlink: {key}")
    print(f"REFERENCE_URLS {len(expected_urls & actual_urls)}/{len(expected_urls)}")
    found_destinations = sum(f"cite.{key}" in destinations for key in keys)
    print(f"REFERENCE_DESTINATIONS {found_destinations}/{len(keys)}")
    found_citations = sum(f"cite.{key}" in actual_citations for key in keys)
    print(f"REFERENCE_CITATIONS {found_citations}/{len(keys)}")
    return errors


def validate_pdf(pdf_path: Path) -> list[str]:
    errors: list[str] = []
    if not pdf_path.exists():
        return [f"compiled PDF does not exist: {pdf_path}"]

    try:
        from pypdf import PdfReader
    except ImportError:
        return ["pypdf is required for PDF validation; install the paper extra"]

    reader = PdfReader(pdf_path)
    pages = len(reader.pages)
    print(f"PAPER_PAGES {pages}/{MAX_PAGES}")
    if pages > MAX_PAGES:
        errors.append(f"PDF has {pages} pages; maximum is {MAX_PAGES}")
    if pages == 0:
        errors.append("PDF has no pages")
        return errors

    for index, page in enumerate(reader.pages, start=1):
        errors.extend(validate_font_resources(page.get("/Resources")))
        width = float(page.mediabox.width)
        height = float(page.mediabox.height)
        if abs(width - US_LETTER_POINTS[0]) > 1 or abs(height - US_LETTER_POINTS[1]) > 1:
            errors.append(
                f"page {index} is {width:g} x {height:g} pt; expected US Letter 612 x 792 pt"
            )
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex", type=Path, default=Path("paper/main.tex"))
    parser.add_argument("--bib", type=Path, default=Path("paper/references.bib"))
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--bib-log", type=Path)
    parser.add_argument(
        "--submission-ready",
        action="store_true",
        help="fail on unresolved markers and missing required AI-text disclosures/citations",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors = validate_sources(args.tex, args.bib, submission_ready=args.submission_ready)
    if args.pdf is not None:
        pdf_errors = validate_pdf(args.pdf)
        errors.extend(pdf_errors)
        if not pdf_errors:
            errors.extend(validate_reference_links(args.pdf, args.bib))
    if args.log is not None:
        errors.extend(validate_log(args.log))
    if args.bib_log is not None:
        errors.extend(validate_bibtex_log(args.bib_log))
    if errors:
        for error in errors:
            print(f"PAPER_GATE_ERROR {error}", file=sys.stderr)
        return 1
    gate = "PAPER_SUBMISSION_GATE_PASS" if args.submission_ready else "PAPER_DRAFT_GATE_PASS"
    print(gate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
