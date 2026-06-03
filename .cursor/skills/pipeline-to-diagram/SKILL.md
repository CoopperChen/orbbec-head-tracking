---
name: pipeline-to-diagram
description: Convert a computer-vision or data-processing pipeline description into a concise Mermaid workflow diagram.
---

# Role
You convert pipeline requirements into clear Mermaid flowcharts that match requested wording and branching logic exactly.

# When To Use
Use this skill when the user asks for:
- a workflow diagram
- a pipeline diagram
- a process flow from text requirements

# Output Rules
- Output only Mermaid flowchart blocks unless the user asks for explanation.
- Preserve user-provided node names and wording exactly where possible.
- Keep diagrams minimal and readable.
- Use decision nodes for explicit conditions (for example: `Face detected?`).
- Use directional flow (`flowchart TD`) by default.
- **Parse-safe labels:** wrap any label that contains parentheses, slashes, colons, or line breaks in double quotes, e.g. `E["Face landmarks - MediaPipe"]`. Avoid raw `<br/>` inside unquoted `[...]`; use a single line or quoted `\n` instead.

# Diagram Template

```mermaid
flowchart TD
    A[Source]
    A --> B[Branch 1]
    A --> C[Branch 2]
    B --> D[Merge]
    C --> D
    D --> E[Processing]
    E --> F{Condition?}
    F -->|Yes| G[Success Output]
    F -->|No| H[Fallback Output]
```

# Example: Face Tracking Pipeline

```mermaid
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
