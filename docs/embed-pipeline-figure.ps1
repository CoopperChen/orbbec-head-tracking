# Embed docs/face-tracking-pipeline-ieee.png into face-tracking-pipeline.md (base64).
# Run from docs/:  .\embed-pipeline-figure.ps1

$pngPath = Join-Path $PSScriptRoot "face-tracking-pipeline-ieee.png"
if (-not (Test-Path $pngPath)) { throw "Missing $pngPath" }

$b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes($pngPath))
$mime = if ([Text.Encoding]::ASCII.GetString([IO.File]::ReadAllBytes($pngPath)[0..2]) -eq "PNG") { "image/png" } else { "image/jpeg" }

$md = @"
# Face Tracking Pipeline

Embedded figure (same pixels as ``face-tracking-pipeline-ieee.png``). Cursor must show this image, not a Mermaid diagram.

<p align="center">
  <img src="data:$mime;base64,$b64" alt="Head pose pipeline" style="max-width:100%; height:auto;" />
</p>

---

| Task | File |
|------|------|
| Edit diagram | ``face-tracking-pipeline-ieee.mmd`` |
| Browser preview | ``face-tracking-pipeline-ieee.html`` |
| Export PNG | ``.\face-tracking-pipeline-export.ps1`` then ``.\embed-pipeline-figure.ps1`` |

> **Fig. 1.** Real-time 6-DoF head pose pipeline (Orbbec Gemini 2L): D2C registration, MediaPipe landmarks, depth-embedded pose, temporal smoothing; null frame if no face.
"@

$out = Join-Path $PSScriptRoot "face-tracking-pipeline.md"
[System.IO.File]::WriteAllText($out, $md)
Write-Host "Wrote $out ($((Get-Item $out).Length) bytes)"
