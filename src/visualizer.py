"""DriveNet live collection monitor — Tesla/Waymo-style split-screen display.

Opens a 1280×720 pygame window with five panels:
  • Front camera       (top-left,     640×480) — primary forward view
  • Semantic BEV       (top-right,    640×480) — top-down semantic seg, z=30 m
  • Left A-pillar cam  (bottom-left,  320×240) — −60° coverage
  • Right A-pillar cam (bottom-center,320×240) — +60° coverage
  • HUD status         (bottom-right, 640×240) — speed, condition, frames

The semantic BEV camera uses CARLA's semantic segmentation sensor mounted
30 m above the ego looking straight down (pitch=−90°).  Each pixel is
colour-coded by label: road=purple, vehicles=blue, pedestrians=red, etc.
A white crosshair marks the ego vehicle at the centre of the BEV.

This module is only imported when enable_viz=True is passed to
DataCollectionAgent.  It has no effect on BC, PPO, or eval stages.

Requirements: pip install pygame
"""
from __future__ import annotations

import queue
import time
import weakref
from typing import Any

import carla
import numpy as np

try:
    import pygame
    _HAS_PYGAME = True
except ImportError:
    _HAS_PYGAME = False

# Capture resolution for viz cameras (lower than training res to save bandwidth)
_CAM_W, _CAM_H = 400, 300
_SEM_W, _SEM_H = 300, 300

# Viz camera transforms match CarlaEnv._CAM_TRANSFORMS exactly
_CAM_TRANSFORMS = [
    carla.Transform(carla.Location(x=1.5, z=2.4)),
    carla.Transform(carla.Location(x=1.0, y=-1.0, z=2.4), carla.Rotation(yaw=-60)),
    carla.Transform(carla.Location(x=1.0, y=1.0,  z=2.4), carla.Rotation(yaw=60)),
]
# Semantic BEV: rooftop-high, pitch=-90 = straight down; FOV=90° gives 60 m coverage
_SEM_TRANSFORM = carla.Transform(
    carla.Location(x=0.0, z=30.0),
    carla.Rotation(pitch=-90.0),
)

# CARLA semantic label → RGB colour (CityScapes palette)
_SEM_PALETTE = np.zeros((256, 3), dtype=np.uint8)
_SEM_PALETTE[0]  = (0,   0,   0)    # Unlabeled
_SEM_PALETTE[1]  = (70,  70,  70)   # Building
_SEM_PALETTE[2]  = (100, 40,  40)   # Fence
_SEM_PALETTE[3]  = (55,  90,  80)   # Other
_SEM_PALETTE[4]  = (220, 20,  60)   # Pedestrian  (red)
_SEM_PALETTE[5]  = (153, 153, 153)  # Pole
_SEM_PALETTE[6]  = (157, 234, 50)   # RoadLine
_SEM_PALETTE[7]  = (128, 64,  128)  # Road        (purple)
_SEM_PALETTE[8]  = (244, 35,  232)  # SideWalk
_SEM_PALETTE[9]  = (107, 142, 35)   # Vegetation
_SEM_PALETTE[10] = (0,   0,   142)  # Vehicles    (blue)
_SEM_PALETTE[11] = (102, 102, 156)  # Wall
_SEM_PALETTE[12] = (220, 220, 0)    # TrafficSign
_SEM_PALETTE[13] = (70,  130, 180)  # Sky
_SEM_PALETTE[14] = (81,  0,   81)   # Ground
_SEM_PALETTE[15] = (150, 100, 100)  # Bridge
_SEM_PALETTE[16] = (230, 150, 140)  # RailTrack
_SEM_PALETTE[17] = (180, 165, 180)  # GuardRail
_SEM_PALETTE[18] = (250, 170, 30)   # TrafficLight
_SEM_PALETTE[19] = (110, 190, 160)  # Static
_SEM_PALETTE[20] = (170, 120, 50)   # Dynamic
_SEM_PALETTE[21] = (45,  60,  150)  # Water
_SEM_PALETTE[22] = (145, 170, 100)  # Terrain

# Window layout (x, y, w, h)
_WIN_W, _WIN_H = 1280, 720
_TOP_H, _BOT_H = 480, 240
_FRONT_RECT = (0,   0,      640, _TOP_H)
_SEM_RECT   = (640, 0,      640, _TOP_H)
_LEFT_RECT  = (0,   _TOP_H, 320, _BOT_H)
_RIGHT_RECT = (320, _TOP_H, 320, _BOT_H)
_HUD_RECT   = (640, _TOP_H, 640, _BOT_H)

_BG     = (10,  10,  15)
_BORDER = (40,  40,  50)
_ACCENT = (0,   200, 255)
_WHITE  = (255, 255, 255)
_GRAY   = (80,  80,  90)
_GREEN  = (0,   200, 80)
_YELLOW = (230, 200, 0)
_RED    = (220, 50,  50)


class DriveNetVisualizer:
    """Real-time collection monitor with multi-camera + semantic BEV display."""

    def __init__(self) -> None:
        if not _HAS_PYGAME:
            raise ImportError(
                "pygame is required for visualization. "
                "Install it with:  pip install pygame"
            )
        pygame.init()
        pygame.display.set_caption("DriveNet — Live Collection View")
        self._screen = pygame.display.set_mode((_WIN_W, _WIN_H))
        self._clock  = pygame.time.Clock()
        self._font_title = pygame.font.SysFont("monospace", 20, bold=True)
        self._font_body  = pygame.font.SysFont("monospace", 18)
        self._font_small = pygame.font.SysFont("monospace", 14)

        self._cam_sensors: list[carla.Actor] = []
        self._cam_queues: list[queue.Queue] = [queue.Queue(maxsize=4) for _ in range(3)]
        self._sem_sensor: carla.Actor | None = None
        self._sem_queue: queue.Queue = queue.Queue(maxsize=4)

        self._cam_frames: list[np.ndarray | None] = [None, None, None]
        self._sem_frame: np.ndarray | None = None

        self._active = True
        self._tick   = 0
        self._render_hz  = 0.0
        self._last_render = time.time()

    # ── Sensor lifecycle ────────────────────────────────────────────────────

    def reattach(self, world: carla.World, vehicle: carla.Vehicle) -> None:
        """Detach sensors from the old vehicle and spawn new ones on *vehicle*.

        Call this immediately after every env.reset() so the viz cameras
        follow the respawned ego vehicle.
        """
        self._detach()
        if not self._active:
            return
        bpl       = world.get_blueprint_library()
        weak_self = weakref.ref(self)

        cam_bp = bpl.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(_CAM_W))
        cam_bp.set_attribute("image_size_y", str(_CAM_H))
        for i, tf in enumerate(_CAM_TRANSFORMS):
            cam = world.spawn_actor(cam_bp, tf, attach_to=vehicle)
            cam.listen(lambda img, qi=i: DriveNetVisualizer._cb_cam(weak_self, img, qi))
            self._cam_sensors.append(cam)

        sem_bp = bpl.find("sensor.camera.semantic_segmentation")
        sem_bp.set_attribute("image_size_x", str(_SEM_W))
        sem_bp.set_attribute("image_size_y", str(_SEM_H))
        sem_bp.set_attribute("fov", "90")
        self._sem_sensor = world.spawn_actor(sem_bp, _SEM_TRANSFORM, attach_to=vehicle)
        self._sem_sensor.listen(lambda img: DriveNetVisualizer._cb_sem(weak_self, img))

    def _detach(self) -> None:
        for sensor in self._cam_sensors:
            try:
                sensor.stop()
                sensor.destroy()
            except Exception:
                pass
        self._cam_sensors.clear()
        for q in self._cam_queues:
            _drain(q)

        if self._sem_sensor is not None:
            try:
                self._sem_sensor.stop()
                self._sem_sensor.destroy()
            except Exception:
                pass
            self._sem_sensor = None
        _drain(self._sem_queue)

    # ── Sensor callbacks (Boost.Asio thread — never raise) ──────────────────

    @staticmethod
    def _cb_cam(weak_self, image, cam_index: int) -> None:
        try:
            self = weak_self()
            if self is not None:
                self._cam_queues[cam_index].put_nowait(image)
        except Exception:
            pass

    @staticmethod
    def _cb_sem(weak_self, image) -> None:
        try:
            self = weak_self()
            if self is not None:
                self._sem_queue.put_nowait(image)
        except Exception:
            pass

    # ── Frame update ────────────────────────────────────────────────────────

    def update(self, info: dict[str, Any] | None = None) -> None:
        """Drain queues and redraw window.  Call once per simulation tick."""
        if not self._active:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._active = False
                return

        self._tick += 1
        if self._tick % 4 != 0:  # render at ~5 Hz
            return

        for i, q in enumerate(self._cam_queues):
            try:
                data = q.get_nowait()
                arr  = np.frombuffer(data.raw_data, dtype=np.uint8)
                arr  = arr.reshape((_CAM_H, _CAM_W, 4))[:, :, :3][:, :, ::-1].copy()
                self._cam_frames[i] = arr
            except (queue.Empty, Exception):
                pass

        try:
            data   = self._sem_queue.get_nowait()
            arr    = np.frombuffer(data.raw_data, dtype=np.uint8).reshape((_SEM_H, _SEM_W, 4))
            labels = arr[:, :, 2]   # semantic tag in R channel of BGRA raw data
            self._sem_frame = _SEM_PALETTE[labels].copy()
        except (queue.Empty, Exception):
            pass

        now = time.time()
        dt  = now - self._last_render
        self._render_hz   = 1.0 / dt if dt > 0.001 else 0.0
        self._last_render = now

        try:
            self._render(info or {})
        except Exception:
            pass  # never let a render error abort collection

    # ── Rendering ───────────────────────────────────────────────────────────

    def _render(self, info: dict[str, Any]) -> None:
        self._screen.fill(_BG)
        self._draw_frame(self._cam_frames[0], *_FRONT_RECT)
        self._draw_frame(self._cam_frames[1], *_LEFT_RECT)
        self._draw_frame(self._cam_frames[2], *_RIGHT_RECT)
        self._draw_bev()
        self._draw_hud(info)
        for rect in (_FRONT_RECT, _SEM_RECT, _LEFT_RECT, _RIGHT_RECT, _HUD_RECT):
            pygame.draw.rect(self._screen, _BORDER, rect, 2)
        self._badge("FRONT  |  60° FOV",  *_FRONT_RECT[:2])
        self._badge("SEMANTIC BEV  |  60 m radius", *_SEM_RECT[:2])
        self._badge("LEFT  −60°", *_LEFT_RECT[:2], small=True)
        self._badge("RIGHT  +60°", *_RIGHT_RECT[:2], small=True)
        pygame.display.flip()
        self._clock.tick(30)

    def _draw_frame(
        self,
        arr: np.ndarray | None,
        x: int, y: int, w: int, h: int,
    ) -> None:
        if arr is None:
            self._screen.blit(
                self._font_small.render("WAITING...", True, _GRAY), (x + w // 2 - 40, y + h // 2)
            )
            return
        surf = pygame.image.frombuffer(arr.tobytes(), (_CAM_W, _CAM_H), "RGB")
        self._screen.blit(pygame.transform.scale(surf, (w, h)), (x, y))

    def _draw_bev(self) -> None:
        x, y, w, h = _SEM_RECT
        sq = min(w, h)  # 480 — square to preserve aspect
        if self._sem_frame is not None:
            surf = pygame.image.frombuffer(
                self._sem_frame.tobytes(), (_SEM_W, _SEM_H), "RGB"
            )
            surf = pygame.transform.scale(surf, (sq, sq))
            ox = x + (w - sq) // 2
            self._screen.blit(surf, (ox, y))
            cx, cy = ox + sq // 2, y + sq // 2
            # Ego-vehicle crosshair at BEV centre
            pygame.draw.circle(self._screen, _WHITE, (cx, cy), 6, 2)
            pygame.draw.line(self._screen, _WHITE, (cx, cy - 14), (cx, cy + 14), 2)
            pygame.draw.line(self._screen, _WHITE, (cx - 14, cy), (cx + 14, cy), 2)

    def _draw_hud(self, info: dict[str, Any]) -> None:
        x0, y0, w, _ = _HUD_RECT
        pad, lh = 10, 24
        x, y = x0 + pad, y0 + pad

        self._screen.blit(
            self._font_title.render(
                f"DriveNet  |  {info.get('town', '---')}", True, _ACCENT
            ), (x, y)
        )
        y += lh + 4
        self._hline(x0, y, w, pad)
        y += 8

        speed = float(info.get("speed_kmh", 0.0))
        sc    = _GREEN if speed < 60 else (_YELLOW if speed < 90 else _RED)
        self._screen.blit(
            self._font_body.render(f"Speed:  {speed:5.1f} km/h", True, sc), (x, y)
        )
        y += lh
        self._bar(x, y, w - pad * 2, 8, speed / 120.0, sc)
        y += 16
        self._hline(x0, y, w, pad)
        y += 8

        cond = info.get("condition", 0)
        self._screen.blit(
            self._font_body.render(f"Cond:   {cond:2d} / 54", True, _WHITE), (x, y)
        )
        y += lh
        for label, key in (("Weather", "weather"), ("Time", "tod"), ("Traffic", "traffic")):
            self._screen.blit(
                self._font_small.render(f"  {label+':':<10}{info.get(key, '---')}", True, _GRAY),
                (x, y)
            )
            y += lh - 4
        y += 4
        self._hline(x0, y, w, pad)
        y += 8

        frames = info.get("frames_saved", 0)
        ftotal = info.get("frames_total", 300)
        chunks = info.get("chunks_saved", 0)
        self._screen.blit(
            self._font_body.render(f"Frames: {frames:4d} / {ftotal}", True, _WHITE), (x, y)
        )
        y += lh
        self._bar(x, y, w - pad * 2, 8, frames / max(ftotal, 1), _GREEN)
        y += 16
        self._screen.blit(
            self._font_body.render(f"Chunks: {chunks:3d} saved", True, _WHITE), (x, y)
        )
        y += lh
        self._hline(x0, y, w, pad)
        y += 8

        hz_col = _GREEN if self._render_hz >= 4.0 else _YELLOW
        self._screen.blit(
            self._font_small.render(
                f"● COLLECTING   Render: {self._render_hz:.1f} Hz", True, hz_col
            ),
            (x, y),
        )

    def _bar(self, x: int, y: int, w: int, h: int, frac: float, color: tuple) -> None:
        frac = max(0.0, min(1.0, frac))
        pygame.draw.rect(self._screen, _GRAY, (x, y, w, h), 1)
        fill = int((w - 2) * frac)
        if fill > 0:
            pygame.draw.rect(self._screen, color, (x + 1, y + 1, fill, h - 2))

    def _hline(self, x0: int, y: int, w: int, pad: int) -> None:
        pygame.draw.line(self._screen, _GRAY, (x0 + pad, y), (x0 + w - pad, y), 1)

    def _badge(self, text: str, x: int, y: int, *, small: bool = False) -> None:
        font = self._font_small if small else self._font_body
        surf = font.render(text, True, _ACCENT)
        bg   = pygame.Surface((surf.get_width() + 8, surf.get_height() + 4), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 160))
        self._screen.blit(bg, (x + 4, y + 4))
        self._screen.blit(surf, (x + 8, y + 6))

    # ── Teardown ────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._active = False
        self._detach()
        pygame.quit()


def _drain(q: queue.Queue) -> None:
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            break
