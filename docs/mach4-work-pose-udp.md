# Mach4 active work pose → Orbbec stream

Publish **active work coordinates** (G54/G55… DRO values, not machine coordinates) from Mach4 to `orbbec-head-stream-cnc` for live pose-aware B/C encoding.

## 1. Mach4 (Lua)

1. Copy [`scripts/mach4_work_pose_publisher.lua`](../scripts/mach4_work_pose_publisher.lua) into your Mach4 profile macros folder, or paste its body into the profile **PLC** script.
2. Set `TARGET_IP` to the Orbbec tracking PC (same host as `--bind-ip` for HICON UDP).
3. Ensure **LuaSocket** is available to Mach4 (`socket.dll` under the Mach4 `api/lua` tree).
4. Call `PublishWorkPoseUdp()` every PLC cycle (default ~200 ms) or from a faster timer.

The script uses `mc.mcAxisGetPos()` — Mach4’s **work-coordinate** position for each axis (X/Y/Z/B/C).

## 2. Orbbec stream PC

```powershell
orbbec-head-stream-cnc `
  --calibration config/cnc_compensation_example.yaml `
  --work-pose-udp-port 62100 `
  --view
```

Priority for nozzle pose used in encoding:

1. Live UDP work pose (if fresh)
2. `--machine-pose`
3. `machine_pose` in calibration YAML

Options:

| Flag | Default | Meaning |
|------|---------|---------|
| `--work-pose-udp-port` | off | Listen for Mach4 JSON on this UDP port |
| `--work-pose-bind-ip` | `0.0.0.0` | Bind address |
| `--work-pose-stale-ms` | `500` | Ignore UDP older than this |
| `--require-work-pose` | off | Do not encode until live work pose arrives |

## 3. JSON packet

```json
{"coord":"work","units":"mm","x":117.44,"y":116.58,"z":-14.2,"b":61.57,"c":20.72}
```

`coord` must be `"work"`. Set `"units":"in"` only if Mach4 is configured in inches.

## 4. Verify

- Mach4 DRO in **work** mode should match published values.
- CNC status panel shows `work pose: live (... ms)` when packets arrive.
- Same panel lists **work X/Y/Z** and **work B/C** (live Mach4 or YAML/`--machine-pose` fallback).
- With `--verbose`, stale/missing pose falls back to static YAML/`--machine-pose`.
