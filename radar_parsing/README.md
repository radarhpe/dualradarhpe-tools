# TI radar processing

Start with an extracted sequence containing the TI capture files under `radar/bin/`. The sequence also needs `radar/timestamps.txt` to align processed frames. Use the supplied timestamps where available. For a single capture block, pass the recording start time in the original recording timezone to regenerate timestamps. Use the dataset timestamps for sequences composed of multiple capture blocks.

```bash
python radar_parsing/repack_timeonly.py -d "$sequence" -st "YYYY-MM-DD HH:MM:SS"
```

## Process raw ADC samples in MATLAB

Open MATLAB (the original workflow used R2022b) and set its current folder to `radar_parsing/mmWave-Matlab`. Run `main.m` and enter the absolute path to the extracted sequence when prompted. Set `process_4d` and `process_pcl` in `main.m` to enable heatmap and point-cloud output, respectively.

The DRAE heatmaps are stored under `radar/mat/` as numbered binary files. The axes represent Doppler, range, azimuth, and elevation.

## Convert heatmaps to Cartesian coordinates

From the repository root, run the Python converter. Its output goes in `radar/npy_DZYX_real/` under the sequence folder by default. Use `-sd` to select another output root.

```bash
python radar_parsing/DRAE_to_DZYX/radar4D2xyz_complex.py -d "$sequence"
```

To rename numbered point-cloud frames using the recording timestamps, run:

```bash
python utils/kitti2time_converter.py -d "$sequence" -t pcl
```
