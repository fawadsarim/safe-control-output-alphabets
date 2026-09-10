param(
    [switch]$SubmissionReady,
    [switch]$ReuseVerifiedArtifacts
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = if ([string]::IsNullOrWhiteSpace($env:SAFE_COMPUTE_PYTHON)) {
    Join-Path $repoRoot ".venv\Scripts\python.exe"
}
else {
    $env:SAFE_COMPUTE_PYTHON
}
$paperDir = Join-Path $repoRoot "paper"
$classDir = Join-Path $repoRoot "artifacts\tooling\ieeeconf"
$classFile = Join-Path $classDir "ieeeconf.cls"
$outputDir = Join-Path $repoRoot "output\pdf"
$outputPdf = Join-Path $outputDir "lcss_acc_2027.pdf"
$expectedClassSha256 = "4befef671c2a996889d325f5170d3387bf42aac9a37dcaa93724ad49816e4ec2"
$templateUrl = "https://css.paperplaza.net/conferences/support/files/ieeeconf.zip"

function Invoke-Checked {
    param([string]$Command, [string[]]$Arguments)
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Command $Arguments"
    }
}

if (-not (Test-Path -LiteralPath $python)) {
    throw "Create .venv and install the pinned dependencies first; see README.md."
}

if (-not (Test-Path -LiteralPath $classFile)) {
    $downloadDir = Join-Path ([System.IO.Path]::GetTempPath()) (
        "lcss-ieeeconf-" + [System.Guid]::NewGuid().ToString("N")
    )
    New-Item -ItemType Directory -Path $downloadDir | Out-Null
    try {
        $archive = Join-Path $downloadDir "ieeeconf.zip"
        Invoke-WebRequest -UseBasicParsing -Uri $templateUrl -OutFile $archive
        Expand-Archive -LiteralPath $archive -DestinationPath $downloadDir
        $downloadedClass = Get-ChildItem -LiteralPath $downloadDir -Filter "ieeeconf.cls" -Recurse |
            Select-Object -First 1
        if ($null -eq $downloadedClass) {
            throw "The official PaperPlaza archive did not contain ieeeconf.cls."
        }
        New-Item -ItemType Directory -Force -Path $classDir | Out-Null
        Copy-Item -LiteralPath $downloadedClass.FullName -Destination $classFile
    }
    finally {
        if (Test-Path -LiteralPath $downloadDir) {
            Remove-Item -LiteralPath $downloadDir -Recurse -Force
        }
    }
}

$actualClassSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $classFile).Hash.ToLowerInvariant()
if ($actualClassSha256 -ne $expectedClassSha256) {
    throw "ieeeconf.cls SHA-256 mismatch: expected $expectedClassSha256, got $actualClassSha256"
}

Push-Location $repoRoot
try {
    if ($ReuseVerifiedArtifacts) {
        Invoke-Checked $python @("scripts/verify_artifacts.py")
    }
    else {
        Invoke-Checked $python @("experiments/pc/generate_information_figure.py")
    }
    Invoke-Checked $python @("scripts/validate_paper.py")
}
finally {
    Pop-Location
}

$oldTexInputs = $env:TEXINPUTS
$oldSourceDateEpoch = $env:SOURCE_DATE_EPOCH
$env:TEXINPUTS = "$classDir;"
$env:SOURCE_DATE_EPOCH = "1788566400"
Push-Location $paperDir
try {
    Invoke-Checked "pdflatex" @("-interaction=nonstopmode", "-halt-on-error", "main.tex")
    Invoke-Checked "bibtex" @("main")
    Invoke-Checked "pdflatex" @("-interaction=nonstopmode", "-halt-on-error", "main.tex")
    Invoke-Checked "pdflatex" @("-interaction=nonstopmode", "-halt-on-error", "main.tex")
}
finally {
    Pop-Location
    $env:TEXINPUTS = $oldTexInputs
    $env:SOURCE_DATE_EPOCH = $oldSourceDateEpoch
}

$validatorArgs = @(
    "scripts/validate_paper.py",
    "--pdf",
    "paper/main.pdf",
    "--log",
    "paper/main.log",
    "--bib-log",
    "paper/main.blg"
)
if ($SubmissionReady) {
    $validatorArgs += "--submission-ready"
}

Push-Location $repoRoot
try {
    Invoke-Checked $python $validatorArgs
    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
    Copy-Item -LiteralPath (Join-Path $paperDir "main.pdf") -Destination $outputPdf -Force
}
finally {
    Pop-Location
}

Write-Output "PAPER_BUILD_PASS output=$outputPdf"
