param(
    [switch]$SubmissionReady,
    [switch]$SkipPaper
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = if ([string]::IsNullOrWhiteSpace($env:SAFE_COMPUTE_PYTHON)) {
    Join-Path $repoRoot ".venv\Scripts\python.exe"
}
else {
    $env:SAFE_COMPUTE_PYTHON
}

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment not found. Follow the pinned setup in README.md."
}

function Invoke-Checked {
    param([string]$Command, [string[]]$Arguments)
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Command $Arguments"
    }
}

$oldPythonPath = $env:PYTHONPATH
$sourcePath = Join-Path $repoRoot "src"
$env:PYTHONPATH = if ([string]::IsNullOrWhiteSpace($oldPythonPath)) {
    $sourcePath
}
else {
    "$sourcePath$([System.IO.Path]::PathSeparator)$oldPythonPath"
}

Push-Location $repoRoot
try {
    New-Item -ItemType Directory -Force -Path (Join-Path $repoRoot "tmp") | Out-Null
    $pytestTemp = "tmp/pytest-reproduce-$PID"
    Invoke-Checked $python @("-m", "ruff", "check", ".")
    Invoke-Checked $python @(
        "-m", "pytest", "-p", "no:cacheprovider", "--basetemp", $pytestTemp
    )
    Invoke-Checked $python @(
        "experiments/pc/information_completion_frontier.py",
        "--output", "artifacts/derived/information_completion_frontier.json"
    )
    Invoke-Checked $python @(
        "experiments/pc/generate_information_figure.py",
        "--input", "artifacts/derived/information_completion_frontier.json"
    )
    Invoke-Checked $python @("scripts/verify_artifacts.py")

    if (-not $SkipPaper) {
        $buildArguments = @(
            "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "scripts/build_paper.ps1",
            "-ReuseVerifiedArtifacts"
        )
        if ($SubmissionReady) {
            $buildArguments += "-SubmissionReady"
        }
        Invoke-Checked "powershell" $buildArguments
    }
}
finally {
    Pop-Location
    $env:PYTHONPATH = $oldPythonPath
}

Write-Output "FULL_REPRODUCTION_PASS"
