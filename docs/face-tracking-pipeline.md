# Face Tracking Pipeline

## Publication figure (orthogonal lines)

![Head pose pipeline](face-tracking-pipeline-orthogonal.svg)

Vector file: `face-tracking-pipeline-orthogonal.svg` (orthogonal connectors, original labels).

---

## Editable Mermaid (your labels)

```mermaid
%%{init: {"flowchart": {"curve": "stepAfter"}}}%%
flowchart TD
    A["Orbbec Gemini 2L"]
    A --> B["RGB frames"]
    A --> C["Depth frames"]
    B --> D["Align depth with RGB"]
    C --> D
    B --> E["Face landmarks detection from RGB - MediaPipe"]
    E --> F{"Face detected?"}
    F -->|Yes| G["Pose solver - depth embedded"]
    D --> G
    G --> H["Smoothing - around previous frame"]
    H --> I["Output: X/Y/Z + Pitch/Yaw/Roll"]
    F -->|No| J["Return no pose frame"]
```

Source file: `face-tracking-pipeline-ieee.mmd`  
Browser: `face-tracking-pipeline-ieee.html`

---

## LaTeX

```latex
\includegraphics[width=\linewidth]{face-tracking-pipeline-orthogonal.pdf}
```
