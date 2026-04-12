param(
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"

$workspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
if (-not $PythonExe) {
    $venvPy = Join-Path $workspaceRoot ".venv\Scripts\python.exe"
    if (Test-Path $venvPy) {
        $PythonExe = $venvPy
    }
    else {
        $PythonExe = "python"
    }
}

$walkforwardScript = Join-Path $workspaceRoot "src\sports\mlb\historical_predictions_mlb_walkforward.py"
$summaryPath = Join-Path $workspaceRoot "src\data\mlb\walkforward\walkforward_summary_mlb.json"
$detailPath = Join-Path $workspaceRoot "src\data\mlb\walkforward\full_game\walkforward_predictions_detail.csv"
$splitsPath = Join-Path $workspaceRoot "src\data\mlb\walkforward\full_game\walkforward_splits_summary.csv"
$outputCsv = Join-Path $workspaceRoot "src\data\mlb\walkforward\prob_shift_sweep_results.csv"

$varsToReset = @(
    "NBA_MLB_MARKETS",
    "NBA_MLB_FULL_GAME_BRIER_WEIGHT",
    "NBA_MLB_META_GATE_ENABLED",
    "NBA_MLB_META_GATE_MODEL_C",
    "NBA_MLB_META_GATE_MIN_CALIB_ROWS",
    "NBA_MLB_META_GATE_MIN_BASE_ROWS",
    "NBA_MLB_META_GATE_THRESHOLD_MIN",
    "NBA_MLB_META_GATE_THRESHOLD_MAX",
    "NBA_MLB_META_GATE_THRESHOLD_STEP",
    "NBA_MLB_META_GATE_MIN_KEEP_ROWS",
    "NBA_MLB_META_GATE_COVERAGE_BONUS",
    "NBA_MLB_META_GATE_RETENTION_TARGET",
    "NBA_MLB_META_GATE_RETENTION_PENALTY",
    "NBA_MLB_META_GATE_MIN_ACC_GAIN",
    "NBA_MLB_META_GATE_MIN_COVERAGE_RETENTION",
    "NBA_MLB_FULL_GAME_PROB_SHIFT_ENABLED",
    "NBA_MLB_FULL_GAME_PROB_SHIFT_MIN",
    "NBA_MLB_FULL_GAME_PROB_SHIFT_MAX",
    "NBA_MLB_FULL_GAME_PROB_SHIFT_STEP"
)

$configs = @(
    [pscustomobject]@{
        Name = "shift_off"
        Env = @{
            NBA_MLB_MARKETS = "full_game"
            NBA_MLB_FULL_GAME_BRIER_WEIGHT = "0.08"
            NBA_MLB_META_GATE_ENABLED = "0"
            NBA_MLB_FULL_GAME_PROB_SHIFT_ENABLED = "0"
            NBA_MLB_FULL_GAME_PROB_SHIFT_MIN = "0.00"
            NBA_MLB_FULL_GAME_PROB_SHIFT_MAX = "0.00"
            NBA_MLB_FULL_GAME_PROB_SHIFT_STEP = "0.01"
        }
    },
    [pscustomobject]@{
        Name = "shift_tight"
        Env = @{
            NBA_MLB_MARKETS = "full_game"
            NBA_MLB_FULL_GAME_BRIER_WEIGHT = "0.08"
            NBA_MLB_META_GATE_ENABLED = "0"
            NBA_MLB_FULL_GAME_PROB_SHIFT_ENABLED = "1"
            NBA_MLB_FULL_GAME_PROB_SHIFT_MIN = "-0.01"
            NBA_MLB_FULL_GAME_PROB_SHIFT_MAX = "0.01"
            NBA_MLB_FULL_GAME_PROB_SHIFT_STEP = "0.01"
        }
    },
    [pscustomobject]@{
        Name = "shift_default"
        Env = @{
            NBA_MLB_MARKETS = "full_game"
            NBA_MLB_FULL_GAME_BRIER_WEIGHT = "0.08"
            NBA_MLB_META_GATE_ENABLED = "0"
            NBA_MLB_FULL_GAME_PROB_SHIFT_ENABLED = "1"
            NBA_MLB_FULL_GAME_PROB_SHIFT_MIN = "-0.04"
            NBA_MLB_FULL_GAME_PROB_SHIFT_MAX = "0.04"
            NBA_MLB_FULL_GAME_PROB_SHIFT_STEP = "0.01"
        }
    },
    [pscustomobject]@{
        Name = "shift_negative"
        Env = @{
            NBA_MLB_MARKETS = "full_game"
            NBA_MLB_FULL_GAME_BRIER_WEIGHT = "0.08"
            NBA_MLB_META_GATE_ENABLED = "0"
            NBA_MLB_FULL_GAME_PROB_SHIFT_ENABLED = "1"
            NBA_MLB_FULL_GAME_PROB_SHIFT_MIN = "-0.08"
            NBA_MLB_FULL_GAME_PROB_SHIFT_MAX = "0.00"
            NBA_MLB_FULL_GAME_PROB_SHIFT_STEP = "0.01"
        }
    },
    [pscustomobject]@{
        Name = "shift_positive"
        Env = @{
            NBA_MLB_MARKETS = "full_game"
            NBA_MLB_FULL_GAME_BRIER_WEIGHT = "0.08"
            NBA_MLB_META_GATE_ENABLED = "0"
            NBA_MLB_FULL_GAME_PROB_SHIFT_ENABLED = "1"
            NBA_MLB_FULL_GAME_PROB_SHIFT_MIN = "0.00"
            NBA_MLB_FULL_GAME_PROB_SHIFT_MAX = "0.08"
            NBA_MLB_FULL_GAME_PROB_SHIFT_STEP = "0.01"
        }
    }
)

$results = New-Object System.Collections.Generic.List[object]
$total = $configs.Count

for ($i = 0; $i -lt $total; $i++) {
    $cfg = $configs[$i]
    $idx = $i + 1
    $pct = [int]((($idx - 1) / $total) * 100)

    Write-Progress -Activity "Barrido prob_shift Full Game" -Status "Ejecutando $idx/$total ($($cfg.Name))" -PercentComplete $pct
    Write-Host "[PROGRESS] START $idx/$total $($cfg.Name)"

    foreach ($name in $varsToReset) {
        Remove-Item "Env:$name" -ErrorAction SilentlyContinue
    }
    foreach ($kv in $cfg.Env.GetEnumerator()) {
        Set-Item "Env:$($kv.Key)" -Value ([string]$kv.Value)
    }

    $logPath = Join-Path $workspaceRoot ("src\data\mlb\walkforward\run_shift_{0}.log" -f $cfg.Name)
    $cmd = '"{0}" "{1}" > "{2}" 2>&1' -f $PythonExe, $walkforwardScript, $logPath
    cmd /d /c $cmd | Out-Null

    if ($LASTEXITCODE -ne 0) {
        Write-Host "[PROGRESS] FAIL  $idx/$total $($cfg.Name) exit=$LASTEXITCODE"
        $results.Add([pscustomobject]@{
                config = $cfg.Name
                global_accuracy_all_games = [double]::NaN
                published_accuracy = [double]::NaN
                coverage = [double]::NaN
                brier = [double]::NaN
                log_loss = [double]::NaN
                published_picks = -1
                meta_gate_on_splits = -1
                prob_shift_mean = [double]::NaN
                prob_shift_min = [double]::NaN
                prob_shift_max = [double]::NaN
                prob_shift_nonzero_splits = -1
                published_confidence_mean = [double]::NaN
                published_confidence_min = [double]::NaN
                published_confidence_max = [double]::NaN
                published_probability_distribution = "run_failed"
                log_file = $logPath
                exit_code = $LASTEXITCODE
            })
        continue
    }

    $summary = Get-Content -Path $summaryPath -Raw | ConvertFrom-Json
    $detail = Import-Csv -Path $detailPath
    $pub = $detail | Where-Object { [int]$_.publish_pick -eq 1 }
    $pubCount = @($pub).Count

    if ($pubCount -gt 0) {
        $probDist = ($pub | Group-Object confidence_bucket | Sort-Object Name | ForEach-Object { "{0}:{1}" -f $_.Name, $_.Count }) -join " | "
        $confAvg = [double](($pub | Measure-Object -Property confidence -Average).Average)
        $confMin = [double](($pub | Measure-Object -Property confidence -Minimum).Minimum)
        $confMax = [double](($pub | Measure-Object -Property confidence -Maximum).Maximum)
    }
    else {
        $probDist = "none"
        $confAvg = 0.0
        $confMin = 0.0
        $confMax = 0.0
    }

    $splits = Import-Csv -Path $splitsPath
    $metaOn = @($splits | Where-Object { [int]$_.meta_gate_enabled -eq 1 }).Count
    $shiftVals = @($splits | ForEach-Object { [double](($_.prob_shift -as [double])) })
    if ($shiftVals.Count -gt 0) {
        $shiftMean = [double](($shiftVals | Measure-Object -Average).Average)
        $shiftMin = [double](($shiftVals | Measure-Object -Minimum).Minimum)
        $shiftMax = [double](($shiftVals | Measure-Object -Maximum).Maximum)
        $shiftNZ = @($shiftVals | Where-Object { [math]::Abs($_) -gt 1e-12 }).Count
    }
    else {
        $shiftMean = 0.0
        $shiftMin = 0.0
        $shiftMax = 0.0
        $shiftNZ = 0
    }

    $results.Add([pscustomobject]@{
            config = $cfg.Name
            global_accuracy_all_games = [double]$summary.full_game.accuracy
            published_accuracy = [double]$summary.full_game.published_accuracy
            coverage = [double]$summary.full_game.published_coverage
            brier = [double]$summary.full_game.brier
            log_loss = [double]$summary.full_game.logloss
            published_picks = [int]$pubCount
            meta_gate_on_splits = [int]$metaOn
            prob_shift_mean = [double]$shiftMean
            prob_shift_min = [double]$shiftMin
            prob_shift_max = [double]$shiftMax
            prob_shift_nonzero_splits = [int]$shiftNZ
            published_confidence_mean = $confAvg
            published_confidence_min = $confMin
            published_confidence_max = $confMax
            published_probability_distribution = $probDist
            log_file = $logPath
            exit_code = 0
        })

    Write-Host ("[PROGRESS] DONE  {0}/{1} {2} acc={3} pub_acc={4} cov={5} picks={6} shift_mean={7} shift_nz={8}" -f $idx, $total, $cfg.Name, [double]$summary.full_game.accuracy, [double]$summary.full_game.published_accuracy, [double]$summary.full_game.published_coverage, $pubCount, $shiftMean, $shiftNZ)
}

Write-Progress -Activity "Barrido prob_shift Full Game" -Completed

$ordered = $results | Sort-Object @{ Expression = "global_accuracy_all_games"; Descending = $true }, @{ Expression = "brier"; Descending = $false }, @{ Expression = "published_accuracy"; Descending = $true }
$ordered | Export-Csv -Path $outputCsv -NoTypeInformation -Encoding UTF8
$ordered | Format-Table -AutoSize | Out-String | Write-Output
Write-Host "Saved: $outputCsv"
