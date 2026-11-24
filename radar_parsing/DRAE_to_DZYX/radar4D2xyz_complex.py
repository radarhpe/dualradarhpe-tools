"""Convert the numbered TI heatmaps in a sequence to Cartesian tensors."""

import argparse
from pathlib import Path
import re

import numpy as np

from radar4D2xyz_complex_python import RTParsing


def main():
    parser = argparse.ArgumentParser(description="Convert DRAE heatmaps to DZYX tensors")
    parser.add_argument("-d", "--dir", type=Path, required=True, help="Sequence directory")
    parser.add_argument("-sd", "--save_dir", type=Path, help="Optional output root")
    parser.add_argument("-st", "--start_frame", type=int, default=0,
                        help="Zero-based starting frame index")
    args = parser.parse_args()

    radar_dir = args.dir / "radar"
    timestamps = np.loadtxt(radar_dir / "timestamps.txt", dtype=str, ndmin=1)
    frame_files = []
    for path in (radar_dir / "mat").glob("4dTensor-Frame_*.bin"):
        match = re.fullmatch(r"4dTensor-Frame_(\d+)\.bin", path.name)
        if match:
            frame_index = int(match[1]) - 1
            if frame_index >= args.start_frame:
                frame_files.append((frame_index, path))
    if not frame_files:
        parser.error("No numbered heatmap files found at or after the starting frame")
    if max(index for index, _ in frame_files) >= len(timestamps):
        parser.error("A heatmap frame number exceeds the number of timestamps")

    output = (args.save_dir or args.dir) / "radar" / "npy_DZYX_real"
    output.mkdir(parents=True, exist_ok=True)
    converter = RTParsing()
    for frame_index, path in sorted(frame_files):
        seconds, _, fraction = timestamps[frame_index].partition(".")
        destination = output / f"{seconds}{fraction.ljust(9, '0')}.npy"
        if destination.exists():
            continue
        heatmap = converter.DRAE_to_DZYX(np.fromfile(path, dtype=np.float32))
        np.save(destination, np.log2(np.maximum(heatmap, 1)).astype(np.float16))


if __name__ == "__main__":
    main()
