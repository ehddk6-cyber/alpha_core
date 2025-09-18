# =================================================================
# Ailey-Bailey's Alpha-Core Data Builder Runner v1.2 (Token Safe Edition)
# =================================================================

# Step 1: Set console output encoding to UTF-8
$OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "Alpha-Core data build will start..."

# --- API Token Environment Variables (leave empty or set externally) ---
# Recommendation: Set tokens in your shell session or Windows user env variables

if (-not $env:EODHD_TOKEN) {
    $env:EODHD_TOKEN = ""
}
if (-not $env:TWELVE_TOKEN) {
    $env:TWELVE_TOKEN = ""
}
if (-not $env:FINNHUB_TOKEN) {
    $env:FINNHUB_TOKEN = ""
}
if (-not $env:FMP_KEY) {
    $env:FMP_KEY = ""
}
if (-not $env:TIINGO_TOKEN) {
    $env:TIINGO_TOKEN = ""
}

# --- Determine Python interpreter ---
$venvPythonPath = ".\.venv\Scripts\python.exe"
$mainScript = "build_alpha_core_data.py"
$pythonExecutable = ""

if (Test-Path $venvPythonPath) {
    Write-Host "Using virtualenv python.exe" -ForegroundColor Cyan
    $pythonExecutable = $venvPythonPath
} else {
    Write-Warning "Virtualenv not found. Using system Python."
    $pythonExecutable = "python"
}

# --- Launch main script ---
if (Test-Path $mainScript) {
    Write-Host "`nRunning '$mainScript'..."
    & $pythonExecutable $mainScript
    $exitCode = $LASTEXITCODE
    if ($exitCode -eq 0) {
        Write-Host "`n[OK] Build finished. Check 'data\alpha_core_data.json'" -ForegroundColor Green
    } else {
        Write-Error "`n[ERROR] Script failed. Exit code: $exitCode"
    }
} else {
    Write-Error "[ERROR] Main script ($mainScript) not found."
}

Write-Host "Build runner finished."
