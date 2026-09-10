from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest


def _load_validator():
    path = Path(__file__).parents[1] / "scripts" / "validate_paper.py"
    spec = importlib.util.spec_from_file_location("validate_paper", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reproduction_bootstraps_ignored_scratch_directory() -> None:
    script = (Path(__file__).parents[1] / "scripts" / "reproduce.ps1").read_text(
        encoding="utf-8"
    )
    create = script.index('New-Item -ItemType Directory -Force -Path (Join-Path $repoRoot "tmp")')
    unique = script.index('$pytestTemp = "tmp/pytest-reproduce-$PID"')
    pytest = script.index('"-m", "pytest"')
    assert create < unique < pytest


def test_reproduction_reuses_frontier_only_after_artifact_verification() -> None:
    root = Path(__file__).parents[1]
    reproduce = (root / "scripts" / "reproduce.ps1").read_text(encoding="utf-8")
    experiment = reproduce.index('"experiments/pc/information_completion_frontier.py"')
    figure = reproduce.index('"experiments/pc/generate_information_figure.py"')
    input_flag = reproduce.index('"--input"', figure)
    verifier = reproduce.index('"scripts/verify_artifacts.py"', input_flag)
    reuse = reproduce.index('"-ReuseVerifiedArtifacts"', verifier)
    assert experiment < figure < input_flag < verifier < reuse

    build = (root / "scripts" / "build_paper.ps1").read_text(encoding="utf-8")
    reuse_branch = build.index("if ($ReuseVerifiedArtifacts)")
    build_verifier = build.index('"scripts/verify_artifacts.py"', reuse_branch)
    source_date = build.index('$env:SOURCE_DATE_EPOCH = "1788566400"')
    latex = build.index('Invoke-Checked "pdflatex"')
    restore_date = build.index("$env:SOURCE_DATE_EPOCH = $oldSourceDateEpoch", latex)
    assert reuse_branch < build_verifier < source_date < latex < restore_date


def _write_sources(tmp_path: Path, *, citation: str = "known", todo: bool = False):
    tex = tmp_path / "main.tex"
    bib = tmp_path / "references.bib"
    marker = (
        "% TODO-AUTHOR-ORCID\n"
        "% TODO-AUTHOR-TELEPHONE\n"
        "% TODO-AUTHOR-POSTAL-ADDRESS\n"
        if todo
        else ""
    )
    tex.write_text(
        "\\documentclass[letterpaper,10pt,conference]{ieeeconf}\n"
        "\\begin{document}\n"
        "\\begin{abstract}A short abstract.\\end{abstract}\n"
        "Language drafting assistance applies throughout. \\cite{openai2026codex}\n"
        f"A cited claim. \\cite{{{citation}}}\n"
        f"{marker}"
        "\\section*{Acknowledgment}\n"
        "OpenAI Codex assisted with manuscript drafting and editing throughout the paper, "
        "code development, and drafting and inspecting proof arguments. The author takes "
        "sole responsibility for the scientific content, analyses, references, and conclusions.\n"
        "\\bibliographystyle{IEEEtran}\n"
        "\\end{document}\n",
        encoding="utf-8",
    )
    bib.write_text(
        "@article{known, title={Known}, url={https://example.org/known}}\n"
        "@misc{openai2026codex, title={Codex}, url={https://openai.com/codex/}}\n",
        encoding="utf-8",
    )
    return tex, bib


def test_valid_draft_sources_pass(tmp_path: Path) -> None:
    validator = _load_validator()
    tex, bib = _write_sources(tmp_path)
    assert validator.validate_sources(tex, bib, submission_ready=False) == []


def test_missing_citation_is_rejected(tmp_path: Path) -> None:
    validator = _load_validator()
    tex, bib = _write_sources(tmp_path, citation="missing")
    assert validator.validate_sources(tex, bib, submission_ready=False) == [
        "missing bibliography keys: missing"
    ]


def test_duplicate_bibliography_key_is_rejected(tmp_path: Path) -> None:
    validator = _load_validator()
    tex, bib = _write_sources(tmp_path)
    bib.write_text(
        bib.read_text(encoding="utf-8") + "@article{known, title={Duplicate}}\n",
        encoding="utf-8",
    )
    assert validator.validate_sources(tex, bib, submission_ready=True) == [
        "duplicate bibliography keys: known"
    ]


def test_submission_gate_rejects_uncited_reference(tmp_path: Path) -> None:
    validator = _load_validator()
    tex, bib = _write_sources(tmp_path)
    bib.write_text(
        bib.read_text(encoding="utf-8") + "@article{unused, title={Unused}}\n",
        encoding="utf-8",
    )
    assert validator.validate_sources(tex, bib, submission_ready=True) == [
        "uncited bibliography keys: unused"
    ]


def test_ai_citation_must_appear_in_manuscript_body(tmp_path: Path) -> None:
    validator = _load_validator()
    tex, bib = _write_sources(tmp_path)
    source = tex.read_text(encoding="utf-8").replace(
        r"Language drafting assistance applies throughout. \cite{openai2026codex}",
        "Language drafting assistance applies throughout.",
    )
    source = source.replace(
        "OpenAI Codex assisted",
        r"\cite{openai2026codex} OpenAI Codex assisted",
    )
    tex.write_text(source, encoding="utf-8")
    assert validator.validate_sources(tex, bib, submission_ready=True) == [
        "AI-assistance citation is missing from the manuscript body"
    ]


@pytest.mark.parametrize(
    "required_scope",
    (
        "OpenAI Codex",
        "manuscript drafting and editing throughout the paper",
        "code development",
        "drafting and inspecting proof arguments",
        "sole responsibility",
    ),
)
def test_ai_disclosure_requires_scope_and_use(tmp_path: Path, required_scope: str) -> None:
    validator = _load_validator()
    tex, bib = _write_sources(tmp_path)
    source = tex.read_text(encoding="utf-8").replace(
        required_scope, "unspecified assistance"
    )
    tex.write_text(source, encoding="utf-8")
    assert validator.validate_sources(tex, bib, submission_ready=True) == [
        "AI-assistance disclosure is missing from Acknowledgment"
    ]


def test_grammar_only_disclosure_understates_this_projects_assistance(tmp_path: Path) -> None:
    validator = _load_validator()
    tex, bib = _write_sources(tmp_path)
    source = tex.read_text(encoding="utf-8")
    prefix, remainder = source.split(r"\section*{Acknowledgment}", maxsplit=1)
    _, suffix = remainder.split(r"\bibliographystyle", maxsplit=1)
    tex.write_text(
        prefix + "\\section*{Acknowledgment}\n"
        "OpenAI Codex improved clarity, grammar, and readability. The author takes sole "
        "responsibility for the scientific content, analyses, references, and conclusions.\n"
        + r"\bibliographystyle" + suffix,
        encoding="utf-8",
    )
    assert validator.validate_sources(tex, bib, submission_ready=True) == [
        "AI-assistance disclosure is missing from Acknowledgment"
    ]


def test_global_disclosure_does_not_replace_section_citations(tmp_path: Path) -> None:
    validator = _load_validator()
    tex, bib = _write_sources(tmp_path)
    source = tex.read_text(encoding="utf-8").replace(
        r"\section*{Acknowledgment}",
        "\\section{Method}\nA method.\n\\section*{Acknowledgment}",
    )
    tex.write_text(source, encoding="utf-8")
    assert validator.validate_sources(tex, bib, submission_ready=True) == [
        "AI-assistance citation is missing from section 1"
    ]
    tex.write_text(
        source.replace("A method.", r"A method.\footnote{Drafting: \cite{openai2026codex}.}"),
        encoding="utf-8",
    )
    assert validator.validate_sources(tex, bib, submission_ready=True) == []


def test_submission_gate_rejects_todo_marker(tmp_path: Path) -> None:
    validator = _load_validator()
    tex, bib = _write_sources(tmp_path, todo=True)
    assert validator.validate_sources(tex, bib, submission_ready=False) == []
    assert validator.validate_sources(tex, bib, submission_ready=True) == [
        "unresolved markers: TODO-AUTHOR-ORCID, TODO-AUTHOR-POSTAL-ADDRESS, "
        "TODO-AUTHOR-TELEPHONE"
    ]


def test_latex_log_rejects_overfull_box(tmp_path: Path) -> None:
    validator = _load_validator()
    clean_log = tmp_path / "clean.log"
    clean_log.write_text("Output written on main.pdf (5 pages).\n", encoding="utf-8")
    assert validator.validate_log(clean_log) == []
    clean_log.write_text("Overfull \\hbox (1.0pt too wide)\n", encoding="utf-8")
    assert validator.validate_log(clean_log) == ["LaTeX log contains overfull horizontal box"]


def test_manuscript_author_block_uses_complete_supplied_data() -> None:
    manuscript = (Path(__file__).parents[1] / "paper" / "main.tex").read_text(
        encoding="utf-8"
    )
    compact = " ".join(manuscript.split())
    for supplied in (
        r"\author{Fawad Sarim",
        "Department of Computer Engineering",
        "COMSATS University Islamabad, Wah Campus",
        "Wah Cantt 47040, Pakistan",
        "Pesgit Engineering Services (SMC-Private) Limited, Rawalpindi 46000, Pakistan",
        "mfawadsarim@gmail.com",
        "ORCID: 0009-0002-8997-9288",
        "Telephone: +923399100007",
        "House no. 1921, Street 32, I-10/2, Islamabad, Pakistan 44000",
    ):
        assert supplied in compact
    assert "Department of Electrical and Computer Engineering" not in compact
    assert "fawad@pesgit.com" not in compact
    assert "TODO-AUTHOR-TELEPHONE" not in manuscript
    assert "TODO-AUTHOR-ORCID" not in manuscript
    assert "TODO-AUTHOR-POSTAL-ADDRESS" not in manuscript


def test_manuscript_links_public_reproduction_code() -> None:
    manuscript = (Path(__file__).parents[1] / "paper" / "main.tex").read_text(
        encoding="utf-8"
    )
    section = manuscript.split(r"\section{Exact Computational Checks}", 1)[1]
    section = section.split(r"\section{Limitations and Conclusion}", 1)[0]
    assert r"\url{https://github.com/fawadsarim/safe-control-output-alphabets}" in section


def test_citation_free_draft_warns_but_cannot_pass_submission_gate(tmp_path, capsys) -> None:
    validator = _load_validator()
    tex, bib = _write_sources(tmp_path)
    source = tex.read_text(encoding="utf-8")
    source = source.replace(r"\cite{openai2026codex}", "")
    source = source.replace("OpenAI Codex assisted", "Assistance was used")
    tex.write_text(source, encoding="utf-8")
    assert validator.validate_sources(tex, bib, submission_ready=False) == []
    assert "L-CSS submission compliance is not met" in capsys.readouterr().out
    assert validator.validate_sources(tex, bib, submission_ready=True) == [
        "uncited bibliography keys: openai2026codex",
        "AI-assistance citation is missing from the manuscript body",
        "AI-assistance disclosure is missing from Acknowledgment",
    ]


def test_manuscript_has_required_disclosures_and_passes_submission_sources(capsys) -> None:
    root = Path(__file__).parents[1]
    tex, bib = root / "paper/main.tex", root / "paper/references.bib"
    validator = _load_validator()
    assert validator.validate_sources(tex, bib, submission_ready=False) == []
    assert "L-CSS submission compliance is not met" not in capsys.readouterr().out
    errors = validator.validate_sources(tex, bib, submission_ready=True)
    assert errors == []


def test_font_gate_rejects_type3_inside_embedded_figure() -> None:
    from pypdf.generic import DictionaryObject, NameObject

    font = DictionaryObject({NameObject("/Subtype"): NameObject("/Type3")})
    nested = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
    })
    form = DictionaryObject({
        NameObject("/Subtype"): NameObject("/Form"), NameObject("/Resources"): nested,
    })
    resources = DictionaryObject({
        NameObject("/XObject"): DictionaryObject({NameObject("/Figure"): form}),
    })
    assert _load_validator().validate_font_resources(resources) == [
        "PDF contains Type 3 font: unnamed"
    ]


def test_font_gate_requires_embedding_and_accepts_embedded_type1() -> None:
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    font = DictionaryObject({NameObject("/Subtype"): NameObject("/Type1")})
    resources = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
    })
    validator = _load_validator()
    assert validator.validate_font_resources(resources) == ["PDF font is not embedded: unnamed"]
    font[NameObject("/FontDescriptor")] = DictionaryObject({
        NameObject("/FontFile"): DecodedStreamObject(),
    })
    assert validator.validate_font_resources(resources) == []


def test_release_figure_has_no_type3_or_unembedded_fonts() -> None:
    from pypdf import PdfReader

    figure = Path(__file__).parents[1] / "paper/figures/information_completion.pdf"
    validator = _load_validator()
    for page in PdfReader(figure).pages:
        assert validator.validate_font_resources(page.get("/Resources")) == []


def test_release_bibliography_has_complete_identifiers_and_citation_coverage() -> None:
    root = Path(__file__).parents[1]
    bib = (root / "paper/references.bib").read_text(encoding="utf-8")
    tex = (root / "paper/main.tex").read_text(encoding="utf-8")
    validator = _load_validator()
    entries = re.findall(r"(?ms)^@\w+\{([^,]+),(.*?)(?=^@|\Z)", bib)
    assert len(entries) == 22
    assert {key for key, _ in entries} == validator._citation_keys(tex)
    for key, fields in entries:
        for field in ("author", "title", "year", "url"):
            assert re.search(rf"\b{field}\s*=\s*\{{[^}}]+\}}", fields), (key, field)
        urls = validator._reference_urls(fields)
        assert len(urls) == 1, key
        doi = re.search(r"\bdoi\s*=\s*\{([^}]+)\}", fields)
        if doi is not None:
            assert urls == {"https://doi.org/" + doi.group(1)}, key
        elif key == "yuceel2026":
            assert urls == {"https://arxiv.org/abs/2604.03132"}
            assert "To appear" in fields
        else:
            assert key == "openai2026codex"
            assert urls == {"https://openai.com/codex/"}


@pytest.mark.parametrize(
    "warning",
    (
        "Warning--empty author in example",
        "Warning--I didn't find a database entry for missing",
        "Repeated entry---line 12 of file references.bib",
    ),
)
def test_bibtex_log_gate_rejects_incomplete_or_duplicate_entries(tmp_path, warning) -> None:
    log = tmp_path / "main.blg"
    log.write_text(warning + "\n", encoding="utf-8")
    assert _load_validator().validate_bibtex_log(log) == [
        "BibTeX log contains a warning or error"
    ]


def test_bibtex_log_gate_accepts_clean_log_and_rejects_missing_log(tmp_path) -> None:
    validator = _load_validator()
    log = tmp_path / "main.blg"
    assert validator.validate_bibtex_log(log) == [f"BibTeX log does not exist: {log}"]
    log.write_text("You've used 22 entries,\n", encoding="utf-8")
    assert validator.validate_bibtex_log(log) == []


@pytest.mark.parametrize("omission", (None, "url", "destination", "citation"))
def test_pdf_gate_checks_actual_reference_links_and_destinations(tmp_path, omission) -> None:
    from pypdf import PdfWriter
    from pypdf.annotations import Link
    from pypdf.generic import DictionaryObject, NameObject, TextStringObject

    _, bib = _write_sources(tmp_path)
    pdf = tmp_path / "references.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    for key, url in (
        ("known", "https://example.org/known"),
        ("openai2026codex", "https://openai.com/codex/"),
    ):
        if not (omission == "url" and key == "known"):
            writer.add_annotation(0, Link(rect=(0, 0, 20, 20), url=url))
        if not (omission == "destination" and key == "known"):
            writer.add_named_destination(f"cite.{key}", 0)
        if not (omission == "citation" and key == "known"):
            citation = Link(rect=(20, 20, 40, 40), url=url)
            citation[NameObject("/A")] = DictionaryObject({
                NameObject("/S"): NameObject("/GoTo"),
                NameObject("/D"): TextStringObject(f"cite.{key}"),
            })
            writer.add_annotation(0, citation)
    writer.write(pdf)
    expected = {
        None: [],
        "url": ["PDF is missing reference hyperlink: https://example.org/known"],
        "destination": ["PDF is missing bibliography destination: known"],
        "citation": ["PDF is missing citation hyperlink: known"],
    }
    assert _load_validator().validate_reference_links(pdf, bib) == expected[omission]
