# Regenerate PNG from cnc-udp-pipeline.mmd via mermaid.ink
# Run: cd docs; .\cnc-udp-pipeline-export.ps1

$mmd = Join-Path $PSScriptRoot "cnc-udp-pipeline.mmd"
$raw = Get-Content $mmd -Raw
$lines = $raw -split "`n" | Where-Object { $_ -notmatch '^\s*%%' -and $_ -notmatch '^\s*---' }
$def = ($lines -join "`n").Trim()
$bytes = [System.Text.Encoding]::UTF8.GetBytes($def)
$b64 = [Convert]::ToBase64String($bytes).TrimEnd('=').Replace('+', '-').Replace('/', '_')
$out = Join-Path $PSScriptRoot "cnc-udp-pipeline.png"
Invoke-WebRequest -Uri "https://mermaid.ink/img/$b64" -OutFile $out -UseBasicParsing
Write-Host "Wrote $out ($((Get-Item $out).Length) bytes)"
