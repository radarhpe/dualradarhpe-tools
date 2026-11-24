# Extracting sensor streams from ROS 2 bags

Install ROS 2 with MCAP support and source the workspace containing the message definitions and converters. The `ros_bag_to_file` launch files below require the sensor-box ROS workspace. The original workspace was referenced as the `dev/sensor_box_rosbag_parsing` branch of [tau-alma/dualRadarHPE-3D](https://github.com/tau-alma/dualRadarHPE-3D). If that workspace is unavailable, use your own ROS 2 subscriber for the listed topics.

Inspect the recording with `ros2 bag info -s mcap "$bag"` before selecting topics. Set `bag` to the path of the recording and `output` to the extracted sequence directory.

For ZED 2i and Bosch streams, start one terminal for playback and one per active converter.

```bash
ros2 bag play -s mcap "$bag"
```

```bash
ros2 launch ros_bag_to_file topic_to_png --ros-args -p output_path:="$output/images" -p input_topic:=/zed2i/zed_node/left/image_rect_color/compressed
ros2 launch ros_bag_to_file topic_to_pcd --ros-args -p output_path:="$output/bosch" -p input_topic:=/off_highway_premium_radar_sample_driver/locations
```

For Ouster scans, use the Ouster replay launch file where available and start the PCD converter in a separate terminal.

```bash
ros2 launch ouster_ros replay.launch.xml bag_file:="$bag"
ros2 launch ros_bag_to_file topic_to_pcd --ros-args -p output_path:="$output/lidar" -p input_topic:=/ouster/points
```
