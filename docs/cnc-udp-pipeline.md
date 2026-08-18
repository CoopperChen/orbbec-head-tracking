# CNC UDP Compensation Pipeline

End-to-end path for `orbbec-head-stream-cnc`: Orbbec Gemini 2L head tracking, XYZBC offset encoding, safety guards, and HICON user-offset streaming over UDP at 100 Hz.

Entry point:

```powershell
orbbec-head-stream-cnc --view
```

## Publication figure (orthogonal, horizontal)

![CNC UDP compensation pipeline](cnc-udp-pipeline-orthogonal.svg)

Vector file: `cnc-udp-pipeline-orthogonal.svg` — left-to-right layout with orthogonal connectors.  
Browser: `cnc-udp-pipeline-orthogonal.html`

---

## Mermaid (horizontal / LR)

```mermaid
%%{init: {"flowchart": {"curve": "stepAfter"}}}%%
flowchart LR
    A["Orbbec Gemini 2L"]
    A --> B["RGB frames"]
    A --> C["Depth frames"]
    B ~~~ C
    B --> D["Align depth with RGB"]
    C --> D
    B --> E["Face landmarks - MediaPipe"]
    E --> F{"Tracking OK?"}
    D --> G["Pose solver - depth rigid"]
    E --> G
    G --> H["Smoothing"]
    H --> I["Encode XYZBC vs baseline"]
    F -->|No| J["Safety hold last offset"]
    F -->|Yes| I
    I --> K["Mismatch and rate limits"]
    J --> K
    K --> L["UDP XYZBC offset output"]
```

Source file: `cnc-udp-pipeline.mmd`  
Mermaid preview: `cnc-udp-pipeline-mermaid.html`  
Orthogonal figure: `cnc-udp-pipeline-orthogonal.html`  
PNG export: `.\cnc-udp-pipeline-export.ps1`

Vision-only stages (without CNC) are documented in `face-tracking-pipeline.md`.
