param(
    [int]$Trials = 2000,
    [int]$Seed = 20260419,
    [int]$TimeoutMinutes = 0,
    [double]$MaxHours = 0,
    [ValidateSet("stable","explore")]
    [string]$SearchMode = "stable",
    [bool]$Resume = $true,
    [bool]$DryRun = $false
)

$ErrorActionPreference = "Stop"

$workspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$pythonExe = Join-Path $workspaceRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python"
}

$scriptPath = Join-Path $workspaceRoot "src\sports\mlb\massive_full_game_search.py"

$argsList = @(
    $scriptPath,
    "--trials", "$Trials",
    "--seed", "$Seed",
    "--python-exe", "$pythonExe",
    "--timeout-minutes", "$TimeoutMinutes",
    "--max-hours", "$MaxHours",
    "--profile", "$SearchMode",
    "--print-every", "1"
)

if ($Resume) { $argsList += "--resume" }
if ($DryRun) { $argsList += "--dry-run" }

Write-Host "Launching massive full_game search..."
Write-Host "Trials=$Trials Seed=$Seed TimeoutMinutes=$TimeoutMinutes MaxHours=$MaxHours Profile=$SearchMode Resume=$Resume DryRun=$DryRun"

& $pythonExe @argsList
