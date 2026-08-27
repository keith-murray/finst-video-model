"""
One-off mp4->gif converter for presentation stimulus samples (no ffmpeg on
this machine) -- reads frames via cv2.VideoCapture, writes an animated gif
via Pillow. Not part of any experiment pipeline; safe to delete after use.

Usage:
    uv run python scripts/prototype/mp4_to_gif.py <in.mp4> <out.gif> [--fps N] [--scale S]
"""

import argparse

import cv2
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_path")
    parser.add_argument("gif_path")
    parser.add_argument("--fps", type=float, default=None, help="output gif fps (default: source fps)")
    parser.add_argument("--scale", type=float, default=1.0, help="resize factor")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    out_fps = args.fps or src_fps

    frames = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if args.scale != 1.0:
            h, w = frame_rgb.shape[:2]
            frame_rgb = cv2.resize(frame_rgb, (int(w * args.scale), int(h * args.scale)))
        frames.append(Image.fromarray(frame_rgb))
    cap.release()

    duration_ms = 1000.0 / out_fps
    frames[0].save(
        args.gif_path, save_all=True, append_images=frames[1:],
        duration=duration_ms, loop=0,
    )
    print(f"Wrote {args.gif_path} ({len(frames)} frames @ {out_fps:.1f}fps)")


if __name__ == "__main__":
    main()
