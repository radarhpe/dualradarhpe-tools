# DualRadarHPE dataset tools

Tools for working with DualRadarHPE, a dataset for 3D human pose estimation using imaging radar. The dataset contains 40 indoor and outdoor sequences with synchronized TI and Bosch radar, stereo camera, LiDAR, and OptiTrack motion capture.

[Project website](https://radarhpe.github.io/) · [Dataset on Fairdata](https://etsin.fairdata.fi/dataset/3493e9d1-1b13-4673-8761-6be372e77af0) · [Paper](https://doi.org/10.1109/OJSP.2026.3711537)

## Quick start

1. Clone or download this repository. From the repository root, create a Python 3.11 environment and install the dependencies.

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   ```

2. Download and extract a sequence from Fairdata. Choose a sequence with TI point clouds and filtered OptiTrack ground truth, such as `civit_easy_01`. See the [sequence index](docs/sequences.md) and [extracted sequence layout](docs/sequence-layout.md).

3. Plot the TI point cloud with its ground truth. Set `sequence` to the absolute path of the extracted sequence directory.

   ```bash
   sequence=/path/to/civit_easy_01
   python visualize/visualize_radarpcl_with_gt.py -d "$sequence" -p radar/radarpcl
   ```

## Workflows

Run these commands from the repository root. Interactive visualization needs a desktop display.

| Task | Command or location |
| --- | --- |
| Plot TI radar point clouds | `python visualize/visualize_pointclouds.py -d "$sequence" --dir_01 radar/radarpcl` |
| Plot TI and Bosch XYZDI point clouds | `python visualize/visualize_pointclouds.py -d "$sequence" --dir_01 radar/radarpcl --dir_02 bosch --calib` |
| Inspect a TI heatmap without annotations | `python visualize/visualize_rtpose_data.py -d "$sequence" --no_gt` |
| Match timestamps from TI point clouds and Bosch XYZDI point clouds | `python calibration/sens2sens_ts_mapping.py -d "$sequence" -r radar/radarpcl -s bosch` |
| Process TI raw ADC data | [Radar processing guide](radar_parsing/README.md) |
| Work with OptiTrack annotations and evaluate predictions | [Annotation scripts](gt_generation_optitrack/) and [evaluation scripts](evaluation/) |

The combined TI/Bosch plot reads a synchronization mapping from `calib/`. Run the timestamp matcher first if the mapping is missing. The timestamp matcher writes its output under the sequence's `calib/` directory. Use `--offset` to shift the secondary sensor timestamps in seconds. Sensor-frame transforms and ZED 2i camera intrinsics are in [calibration/extrinsic_calib.py](calibration/extrinsic_calib.py).

Scripts that read PCD files with Open3D also require `python -m pip install open3d`. TI raw ADC processing requires MATLAB. Extracting camera, Bosch radar, and LiDAR streams from MCAP bags requires ROS 2 and the corresponding message packages. The [bag extraction notes](docs/rosbags.md) list the topics and commands.

## Credits

**Dataset and paper authors** — Trung Nguyen, Prashant Kumar Rai, Nikolay Serbenyuk, Reza Ghabcheloo, and Nataliya Strokina.

**Code contributors** — Trung Nguyen contributed the heatmap converter and pose visualization scripts. Prashant Kumar Rai adapted the TI radar timestamp extraction script.

Upstream contributors are credited in the [third-party notices](THIRD_PARTY_NOTICES.md).

## Citation

If you use DualRadarHPE in research, cite the dataset paper.

```bibtex
@article{nguyen2026dualradarhpe,
  author  = {Nguyen, Trung and Rai, Prashant Kumar and Serbenyuk, Nikolay and
             Ghabcheloo, Reza and Strokina, Nataliya},
  title   = {A Multi-Sensor Dataset for 3D Human Pose Estimation Using Imaging Radar},
  journal = {IEEE Open Journal of Signal Processing},
  year    = {2026},
  volume  = {7},
  pages   = {753--765},
  doi     = {10.1109/OJSP.2026.3711537}
}
```

## Terms and attribution

See the Fairdata record for dataset access and usage terms. See [third-party notices](THIRD_PARTY_NOTICES.md) for bundled upstream code.
