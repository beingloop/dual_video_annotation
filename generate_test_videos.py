import math
import os
from pathlib import Path
from typing import Callable, Tuple

import cv2
import numpy as np


DurationSeconds = 300
FramesPerSecond = 24
FrameSize: Tuple[int, int] = (640, 360)
OutputRoots = [Path("video1"), Path("video2")]


def ensure_output_dirs() -> None:
    for root in OutputRoots:
        root.mkdir(parents=True, exist_ok=True)


def generate_video(
    output_path: Path,
    frame_generator: Callable[[int, int], np.ndarray],
    duration: int = DurationSeconds,
    fps: int = FramesPerSecond,
    frame_size: Tuple[int, int] = FrameSize,
) -> None:
    total_frames = duration * fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)

    if not writer.isOpened():
        raise RuntimeError(f"无法打开视频写入器: {output_path}")

    for frame_idx in range(total_frames):
        frame = frame_generator(frame_idx, total_frames)
        writer.write(frame)

    writer.release()


def clip_value(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def pattern_primary(frame_idx: int, total_frames: int) -> np.ndarray:
    height, width = FrameSize[1], FrameSize[0]
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # 背景渐变随时间缓慢变化
    t = frame_idx / total_frames
    gradient = np.linspace(0, 255, width, dtype=np.uint8)
    gradient = np.tile(gradient, (height, 1))
    shift = int((math.sin(2 * math.pi * t) + 1) * 50)
    canvas[..., 0] = np.roll(gradient, shift, axis=1)
    canvas[..., 1] = np.roll(gradient, shift // 2, axis=1)
    canvas[..., 2] = np.roll(gradient, shift // 3, axis=1)

    # 绘制旋转矩形
    center = (width // 2, height // 2)
    base_size = 180
    for i in range(3):
        angle = (frame_idx * (0.4 + 0.1 * i)) % 360
        scale = 0.6 + 0.2 * math.sin(2 * math.pi * t * (i + 1))
        rect_size = (int(base_size * scale * (1 + 0.2 * i)), int(base_size * scale * (1 - 0.1 * i)))
        color = (
            int(clip_value(180 + 70 * math.sin(angle / 30 + i), 0, 255)),
            int(clip_value(120 + 70 * math.cos(angle / 40 + i), 0, 255)),
            int(clip_value(200 + 40 * math.sin(angle / 25 + i), 0, 255)),
        )
        draw_rotated_rectangle(canvas, center, rect_size, angle, color, thickness=6)

    return canvas


def pattern_secondary(frame_idx: int, total_frames: int) -> np.ndarray:
    height, width = FrameSize[1], FrameSize[0]
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    t = frame_idx / FramesPerSecond
    # 背景添加水平波纹
    for y in range(height):
        intensity = int(120 + 80 * math.sin(2 * math.pi * (y / 80) + t / 3))
        canvas[y, :, :] = (intensity, intensity, intensity)

    # 绘制多彩圆环依时间扩散
    num_circles = 7
    max_radius = min(width, height) // 2 - 20
    center = (width // 2, height // 2)

    for i in range(num_circles):
        phase = (t / 2 + i * 0.4)
        radius = int(max_radius * (0.2 + 0.8 * ((math.sin(phase) + 1) / 2)))
        color = (
            int(clip_value(80 + 160 * ((math.sin(phase * 1.3) + 1) / 2), 0, 255)),
            int(clip_value(60 + 160 * ((math.cos(phase * 1.7) + 1) / 2), 0, 255)),
            int(clip_value(140 + 100 * ((math.sin(phase * 0.9) + 1) / 2), 0, 255)),
        )
        cv2.circle(canvas, center, radius, color, thickness=8)

    # 绘制交错跳动的小圆点
    dot_count = 40
    for idx in range(dot_count):
        angle = 2 * math.pi * idx / dot_count
        radius = 40 + 20 * math.sin(t * 2 + idx / 5)
        x = int(center[0] + (radius + 100) * math.cos(angle + t / 4))
        y = int(center[1] + (radius + 100) * math.sin(angle + t / 5))
        color = (
            int(clip_value(200 + 40 * math.sin(angle * 2 + t), 0, 255)),
            int(clip_value(160 + 60 * math.cos(angle * 3 + t / 2), 0, 255)),
            int(clip_value(220 + 30 * math.sin(angle * 4 + t / 3), 0, 255)),
        )
        cv2.circle(canvas, (x, y), 6, color, thickness=-1)

    return canvas


def draw_rotated_rectangle(
    image: np.ndarray,
    center: Tuple[int, int],
    size: Tuple[int, int],
    angle: float,
    color: Tuple[int, int, int],
    thickness: int = 3,
) -> None:
    # 绘制旋转矩形，使用 OpenCV 多边形避免额外转换
    angle_rad = math.radians(angle)
    b = size[0] / 2
    c = size[1] / 2

    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    points = [
        (
            int(center[0] + cos_a * b - sin_a * c),
            int(center[1] + sin_a * b + cos_a * c),
        ),
        (
            int(center[0] - cos_a * b - sin_a * c),
            int(center[1] - sin_a * b + cos_a * c),
        ),
        (
            int(center[0] - cos_a * b + sin_a * c),
            int(center[1] - sin_a * b - cos_a * c),
        ),
        (
            int(center[0] + cos_a * b + sin_a * c),
            int(center[1] + sin_a * b - cos_a * c),
        ),
    ]

    pts = np.array(points, dtype=np.int32)
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)


if __name__ == "__main__":
    ensure_output_dirs()

    output_a = OutputRoots[0] / "sample_primary.mp4"
    output_b = OutputRoots[1] / "sample_secondary.mp4"

    generate_video(output_a, pattern_primary)
    generate_video(output_b, pattern_secondary)

    print("测试视频生成完成:")
    print(f"  主视角: {output_a}")
    print(f"  辅视角: {output_b}")
    print("可将两个文件夹设为主辅视角，进行标注流程演练。")
