# Regenerate PNG from face-tracking-pipeline-ieee.mmd via mermaid.ink
# Run: cd docs; .\face-tracking-pipeline-export.ps1

$mmd = Join-Path $PSScriptRoot "face-tracking-pipeline-ieee.mmd"
$raw = Get-Content $mmd -Raw
# Strip YAML/comment lines; mermaid.ink needs diagram body only
$lines = $raw -split "`n" | Where-Object { $_ -notmatch '^\s*%%' -and $_ -notmatch '^\s*---' }
$def = ($lines -join "`n").Trim()
$bytes = [System.Text.Encoding]::UTF8.GetBytes($def)
$b64 = [Convert]::ToBase64String($bytes).TrimEnd('=').Replace('+', '-').Replace('/', '_')
$out = Join-Path $PSScriptRoot "face-tracking-pipeline-ieee.png"
Invoke-WebRequest -Uri "https://mermaid.ink/img/$b64" -OutFile $out -UseBasicParsing
Write-Host "Wrote $out ($((Get-Item $out).Length) bytes)"
