from __future__ import annotations

import argparse
import math
import sys
import time
import cv2
import numpy as np

from fluid_sim import FluidParams, StableFluid
from hand_tracking import HandInfo, HandTracker

WINDOW_TITLE = "Fluid Particle Simulation"
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FLUID_QUAD = np.array([[280, 460], [620, 420], [580, 180], [260, 220]], dtype=np.float32)
SIM_PANEL_LINES = [
    "radius      0.052",
    "dfriction   0.0",
    "sfriction   0.0",
    "pfriction   0.0",
    "rest        0.088",
    "adhesion    0.0",
    "sleepthresh 0.0",
    "clampspeed  0.0",
    "maxspeed    100.0",
    "clampaccel  0.1",
    "maxaccel    100.0",
    "diss        0.0",
    "damping     0.059",
    "cohesion    0.336",
    "surftension 0.03",
    "viscosity   0.001",
    "buoyancy    0.0",
    "collidist   0.1",
    "scrollmargin 0.1",
    "smoothing   0.0",
    "vortconf    90.0",
]

_VIGNETTE_CACHE: dict[tuple[int, int], np.ndarray] = {}
BASE_CENTER = FLUID_QUAD.mean(axis=0)
CUBE_WIDTH = 360.0
CUBE_HEIGHT = 260.0
CUBE_DEPTH = 240.0
CAMERA_DIST = 900.0
FOCAL_LENGTH = 780.0
_HALF_W = CUBE_WIDTH / 2.0
_HALF_H = CUBE_HEIGHT / 2.0
CUBE_POINTS = np.array(
    [
        [-_HALF_W, _HALF_H, 0.0],
        [_HALF_W, _HALF_H, 0.0],
        [_HALF_W, -_HALF_H, 0.0],
        [-_HALF_W, -_HALF_H, 0.0],
        [-_HALF_W, _HALF_H, CUBE_DEPTH],
        [_HALF_W, _HALF_H, CUBE_DEPTH],
        [_HALF_W, -_HALF_H, CUBE_DEPTH],
        [-_HALF_W, -_HALF_H, CUBE_DEPTH],
    ],
    dtype=np.float32,
)
FRONT_FACE_IDX = np.array([0, 1, 2, 3], dtype=np.int32)
CUBE_EDGES = np.array(
    [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ],
    dtype=np.int32,
)

cv2.setUseOptimized(True)


def normalize_density(sim: StableFluid) -> np.ndarray:
    density = sim.density[1 : sim.params.n + 1, 1 : sim.params.n + 1]
    norm = cv2.normalize(density, None, 0, 255, cv2.NORM_MINMAX)
    return norm.astype(np.uint8)


def make_fluid_texture(sim: StableFluid, size: int = 256) -> np.ndarray:
    norm = normalize_density(sim)
    resized = cv2.resize(norm, (size, size), interpolation=cv2.INTER_CUBIC)
    colored = cv2.applyColorMap(resized, cv2.COLORMAP_HOT)
    alpha = cv2.GaussianBlur(resized, (0, 0), 3)
    alpha = cv2.normalize(alpha, None, 0, 255, cv2.NORM_MINMAX)
    texture = np.dstack((colored, alpha))
    return texture


def composite_texture(canvas: np.ndarray, texture: np.ndarray, quad: np.ndarray) -> np.ndarray:
    h, w, _ = canvas.shape
    src = np.float32([[0, 0], [texture.shape[1], 0], [texture.shape[1], texture.shape[0]], [0, texture.shape[0]]])
    dst = quad.astype(np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(texture, matrix, (w, h))
    color = warped[:, :, :3]
    alpha = warped[:, :, 3:] / 255.0
    canvas = (color * alpha + canvas * (1 - alpha)).astype(np.uint8)
    return canvas


def draw_cube_edges(canvas: np.ndarray, projected_pts: np.ndarray) -> None:
    pts = projected_pts.astype(np.int32)
    front = pts[FRONT_FACE_IDX]
    back = pts[FRONT_FACE_IDX + 4]
    for poly in (front, back):
        cv2.polylines(canvas, [poly], True, (20, 12, 8), 5, cv2.LINE_AA)
        cv2.polylines(canvas, [poly], True, (140, 210, 255), 2, cv2.LINE_AA)
    for start, end in CUBE_EDGES:
        p0 = tuple(pts[start])
        p1 = tuple(pts[end])
        cv2.line(canvas, p0, p1, (5, 5, 5), 7, cv2.LINE_AA)
        cv2.line(canvas, p0, p1, (60, 110, 210), 4, cv2.LINE_AA)
        cv2.line(canvas, p0, p1, (255, 255, 255), 1, cv2.LINE_AA)
    for idx in range(len(pts)):
        cv2.circle(canvas, tuple(pts[idx]), 3, (255, 240, 180), -1, cv2.LINE_AA)


def project_cube(
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    translate: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float32)
    rot_z = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    rot = rot_z @ rot_y @ rot_x

    points = (CUBE_POINTS @ rot.T) * scale
    z_cam = points[:, 2] + CAMERA_DIST
    persp = (FOCAL_LENGTH / z_cam)[:, None]
    projected = points[:, :2] * persp + BASE_CENTER + translate
    front_quad = projected[FRONT_FACE_IDX]
    return front_quad.astype(np.float32), projected.astype(np.float32)


def draw_text_block(canvas: np.ndarray, origin: tuple[int, int], lines: list[str], scale: float = 0.6) -> None:
    x, y = origin
    for line in lines:
        cv2.putText(
            canvas,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += int(26 * scale)


def map_finger_to_grid(sim: StableFluid, px: int, py: int, frame_shape: tuple[int, int, int]) -> tuple[int, int]:
    h, w, _ = frame_shape
    gx = int(np.interp(px, [0, w], [1, sim.params.n]))
    gy = int(np.interp(py, [0, h], [1, sim.params.n]))
    return gx, gy


def update_sim_with_hand(
    sim: StableFluid,
    hand: HandInfo,
    frame_shape: tuple[int, int, int],
) -> float:
    gx, gy = map_finger_to_grid(sim, hand.index_tip[0], hand.index_tip[1], frame_shape)
    injection = 90 + 160 * hand.pinch_strength
    speed = math.hypot(hand.velocity[0], hand.velocity[1])
    radius = int(np.clip(2 + speed / 220.0, 2, 7))
    sim.add_density(gx, gy, injection, radius=radius)
    vel_scale = 0.0045
    sim.add_velocity(gx, gy, hand.velocity[0] * vel_scale, hand.velocity[1] * vel_scale, radius=radius + 1)
    return speed


def stylize_background(frame: np.ndarray) -> np.ndarray:
    dark = cv2.addWeighted(frame, 0.4, np.zeros_like(frame), 0.6, 0)
    blur = cv2.GaussianBlur(dark, (0, 0), 1.1)
    return cv2.addWeighted(dark, 1.8, blur, -0.8, 0)


def sharpen_frame(image: np.ndarray) -> np.ndarray:
    soft = cv2.GaussianBlur(image, (0, 0), 0.8)
    sharpened = cv2.addWeighted(image, 1.6, soft, -0.6, 0)
    return sharpened


def apply_vignette(image: np.ndarray) -> np.ndarray:
    key = (image.shape[1], image.shape[0])
    mask = _VIGNETTE_CACHE.get(key)
    if mask is None:
        kernel_x = cv2.getGaussianKernel(key[0], key[0] / 2.0)
        kernel_y = cv2.getGaussianKernel(key[1], key[1] / 2.0)
        kernel = kernel_y @ kernel_x.T
        mask = cv2.normalize(kernel, None, 0.4, 1.0, cv2.NORM_MINMAX).astype(np.float32)
        _VIGNETTE_CACHE[key] = mask
    vignette = (image.astype(np.float32) * mask[..., None]).astype(np.uint8)
    return vignette


def format_param_lines(params: FluidParams, energy: float) -> list[str]:
    lines = ["Fluid Particle Simulation"]
    lines.extend(SIM_PANEL_LINES)
    lines.append(f"energy      {energy:4.2f}")
    lines.append(f"dt          {params.dt:.3f}")
    lines.append(f"diffusion   {params.diffusion:.6f}")
    lines.append(f"viscosity   {params.viscosity:.6f}")
    lines.append(f"dissipation {params.dissipation:.3f}")
    return lines


def format_hand_lines(hand: HandInfo | None, fps: float, frame_shape: tuple[int, int, int]) -> list[str]:
    if not hand:
        return [
            "+ HAND_1_ROTATION: --.-",
            "+ HAND_1_POSITION_X: --.-",
            "+ HAND_1_POSITION_Y: --.-",
            "+ HAND_1_POSITION_Z: --.-",
            f"+ FPS:{fps:4.1f}",
        ]
    h, w, _ = frame_shape
    rotation = math.degrees(math.atan2(hand.velocity[1], hand.velocity[0]))
    norm_x = (hand.index_tip[0] / w) - 0.5
    norm_y = (hand.index_tip[1] / h) - 0.5
    depth = (0.5 - hand.pinch_strength) * 0.2
    return [
        f"+ HAND_1_ROTATION:{rotation:5.1f}",
        f"+ HAND_1_POSITION_X:{norm_x:+0.2f}",
        f"+ HAND_1_POSITION_Y:{-norm_y:+0.2f}",
        f"+ HAND_1_POSITION_Z:{depth:+0.2f}",
        f"+ FPS:{fps:4.1f}",
    ]


def run(camera_index: int) -> None:
    params = FluidParams()
    sim = StableFluid(params)
    sim.warmup(2)
    tracker = HandTracker()
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Unable to access webcam")
    last_time = time.time()
    fps_smooth = 0.0
    hand_speed = 0.0
    cube_angle = 0.0
    cube_translate = np.zeros(2, dtype=np.float32)
    cube_pitch = 0.0
    cube_roll = 0.0
    cube_scale = 1.0

    text_headline = "playing with fluid particle stimulation"

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        hand = tracker.detect(frame)
        if hand:
            hand_speed = update_sim_with_hand(sim, hand, frame.shape)
            norm_x = hand.index_tip[0] / FRAME_WIDTH - 0.5
            norm_y = hand.index_tip[1] / FRAME_HEIGHT - 0.5
            vel = np.clip(np.array(hand.velocity, dtype=np.float32), -900.0, 900.0)
            target_translate = np.array(
                [norm_x * 160.0 + vel[0] * 0.035, norm_y * 90.0 + vel[1] * 0.02],
                dtype=np.float32,
            )
            cube_translate = cube_translate * 0.82 + target_translate * 0.18
            target_angle = np.clip(norm_x * 42.0 + vel[0] * 0.02, -65.0, 65.0)
            cube_angle = cube_angle * 0.85 + target_angle * 0.15
            target_pitch = np.clip(-norm_y * 55.0 + vel[1] * 0.015, -65.0, 65.0)
            cube_pitch = cube_pitch * 0.85 + target_pitch * 0.15
            twist = (hand.pinch_strength - 0.5) * 70.0 + vel[0] * 0.01 - vel[1] * 0.008
            target_roll = np.clip(twist, -40.0, 40.0)
            cube_roll = cube_roll * 0.85 + target_roll * 0.15
            target_scale = np.clip(1.0 + (hand.pinch_strength - 0.5) * 0.25, 0.85, 1.2)
            cube_scale = cube_scale * 0.9 + target_scale * 0.1
        else:
            hand_speed *= 0.85
            cube_translate *= 0.9
            cube_angle *= 0.92
            cube_pitch *= 0.92
            cube_roll *= 0.92
            cube_scale = cube_scale * 0.95 + 1.0 * 0.05
        sim.step()
        fluid_energy = float(np.clip(sim.density.mean() * 0.02 + hand_speed * 0.0005, 0.0, 1.2))

        background = stylize_background(frame)
        background = apply_vignette(background)
        texture = make_fluid_texture(sim)
        cube_quad, cube_pts = project_cube(cube_angle, cube_pitch, cube_roll, cube_translate, cube_scale)
        background = composite_texture(background, texture, cube_quad)
        draw_cube_edges(background, cube_pts)

        cv2.putText(
            background,
            text_headline,
            (80, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        draw_text_block(background, (80, 150), format_param_lines(params, fluid_energy))
        now = time.time()
        inst_fps = 1.0 / max(1e-4, now - last_time)
        if fps_smooth == 0.0:
            fps_smooth = inst_fps
        else:
            fps_smooth = fps_smooth * 0.9 + inst_fps * 0.1
        last_time = now
        draw_text_block(background, (950, 150), format_hand_lines(hand, fps_smooth, frame.shape))

        background = sharpen_frame(background)
        cv2.imshow(WINDOW_TITLE, background)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
    cap.release()
    cv2.destroyAllWindows()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive fluid particle demo")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    run(args.camera)


if __name__ == "__main__":
    main()
