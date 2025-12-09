# Fluid Particle Simulation Demo

A Python + OpenCV recreation of the "playing with fluid particle stimulation" experiment. Built by `tubakhxn`, it opens your webcam, tracks one hand with MediaPipe, and injects energy into a 2D stable-fluid solver so it feels like you are stirring a cube filled with glowing fluid.

## Features
- Real-time fluid solver (Jos Stam "Stable Fluids") implemented from scratch in `numpy` and JIT-accelerated with `numba` for lower latency.
- MediaPipe hand tracking; pinch strength controls how much density is injected, finger velocity pushes the fluid.
- Stylised overlay that mimics the reference look with cube projection, HUD text, and live FPS.

## Requirements
- Python 3.10+
- Webcam accessible from the host OS

Install dependencies with:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Running the demo

```powershell
python main.py
```

Optional arguments:

- `--camera <index>` – choose a different webcam index if you have multiple cameras.

## Controls
- Hold a hand in the camera view. The index finger tip drives the injection point.
- Pinch your thumb and index finger to increase the amount of glowing fluid being emitted.
- Move quickly to fling the particles across the cube; velocity vectors are derived from the finger motion between frames.
- Press `q` or `Esc` to exit.

## File overview
- `fluid_sim.py` – grid-based stable fluid solver.
- `hand_tracking.py` – MediaPipe wrapper returning pinch info + finger coordinates.
- `main.py` – orchestrates webcam capture, draws the UI, and couples the hand input to the solver.

## Credit & forking guidelines
- Creator: **tubakhxn**. Please retain attribution when showcasing or remixing the project.
- To fork: duplicate the repo on GitHub (Fork button), clone your fork, create a feature branch, and submit a pull request back to the original `tubakhxn` repository when you have improvements to share.
