import numpy as np
import os
import os.path as osp
import argparse
import re

# npy_DZYX_path_tail = "radar/npy_DZYX_real"
# mapping_file_tail = "calib/radarpcl2gt_mapping.txt"

def main():
    parser = argparse.ArgumentParser(
        prog="kitti2time.py",
        description="For converting radar frame between kitti format (index) and timestamp"
    )

    parser.add_argument(
        "-d", "--dir",
        type=str,
        help="Path to data dir"
    )

    parser.add_argument(
        "-t", "--type",
        choices=("radarpcl", "pcl", "4d", "4D"),
        default="radarpcl",
        help="Data type of the frame (radarpcl/4d)"
    )

    # parser.add_argument(
    #     "-m", "--mode",
    #     type=str,
    #     default="k2t",
    #     help="Choose mode to convert kitti to timestamp (k2t), vice versa"
    # )

    args = parser.parse_args()

    radar_dir = osp.join(args.dir, "radar")
    is_point_cloud = args.type in ("radarpcl", "pcl")
    frame_dir = osp.join(radar_dir, "radarpcl" if is_point_cloud else "npy_DZYX_real")
    timestamps = np.loadtxt(osp.join(radar_dir, "timestamps.txt"), dtype=str, ndmin=1)

    if is_point_cloud:
        frames = []
        for name in sorted(os.listdir(frame_dir)):
            match = re.fullmatch(r"radarpcl_(\d+)\.xyzdi", name)
            if match:
                index = int(match[1]) - 1
                if not 0 <= index < len(timestamps):
                    parser.error(f"No timestamp for {name}")
                frames.append((timestamps[index], name))
    else:
        frames = zip(timestamps, sorted(name for name in os.listdir(frame_dir)
                                        if name.endswith(".npy")))

    for timestamp, name in frames:
        seconds, _, fraction = timestamp.partition(".")
        extension = osp.splitext(name)[1]
        target = osp.join(frame_dir, f"{seconds}{fraction.ljust(9, '0')}{extension}")
        source = osp.join(frame_dir, name)
        if source != target and osp.exists(target):
            parser.error(f"Output already exists: {target}")
        os.rename(source, target)


if __name__ == "__main__":
    main()
