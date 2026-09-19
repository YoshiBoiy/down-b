# Down-B

Down-B is an adaptive virtual boxing opponent. The current default demo runs on a laptop webcam in the browser: it uses MediaPipe Pose Landmarker for live pose tracking and Three.js to render a 3D mesh-style boxer plus virtual opponent. The repo also keeps the RDK X5/stereo-camera architecture as swappable backend modules.

The code runs today with a deterministic simulator, then swaps in RDK capture and BPU inference adapters on device.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
down-b-sim --web-root web
```

Open `http://localhost:8765` after starting the server, then allow camera access. Use Chrome or Edge for the most reliable webcam/GPU path.

The browser loads MediaPipe and Three.js from CDNs, so the first webcam run needs internet access. Camera processing happens locally in your browser; the server is only serving files.

## Project map

```text
down_b/
  capture/          synchronized stereo frame interfaces and calibration
  inference/        pose estimator interface, mock estimator, player lock
  geometry/         disparity sampling and 3D reprojection
  tracking/         body state schema and temporal filters
  boxing/           action recognition, guard, hit model, adaptation, opponent
  api/              WebSocket/static-file server
  storage/          session metadata, metrics, and event JSONL writer
  simulation/       deterministic body-state stream for local testing
web/                laptop-camera pose tracker and 3D mesh opponent visualization
config/             camera, threshold, and opponent defaults
tests/              unit coverage for geometry, actions, hits, adaptation
```

## RDK integration path

1. Populate `config/camera.yaml` with calibrated stereo intrinsics/extrinsics.
2. Replace `MockPoseEstimator` with `PoseEstimator` backed by `hbm_runtime`.
3. Feed `StereoFrame` instances from ROS2 dual MIPI topics.
4. Keep the game loop unchanged: it consumes filtered `BodyState` objects and publishes structured frame/game events.
