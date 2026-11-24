# Extracted sequence layout

```
seq_name # sequence root dir
+-- bosch   # radar pcl from Bosch OHM radar (Optional)
    +-- <timestamp>.xyzdi
        # radar point cloud array from Bosch OHM radar
        # Format: <Point_x>,<Point_y>,<Point_z>,<Doppler>,<RadarCrossSection>
    +-- ....

+-- bosch_mars # data tensor for MARS/PointMLP/RadarHPE_ViT (Optional)
    +-- featuremap_<seq_name>_<type>.npy
        # MARS type feature map of Bosch radar point cloud
        # Format: 4D Tensor with shape (n_frames, 12, 12, 5)
        #         with (12, 12, 5) is 144 points, sorted using their x and y values
        #         each point has 5 values (x, y, z, Doppler, RadarCrossSection)

    +-- labels_<seq_name>_<type>.npy
        # MARS type ground truth in Bosch radar coordinate
        # Format: 3D Tensor with shape (n_frames, 15, 3)
        #         with 15 is the number of joints and 3 is for (x, y, z)

    +-- good_idx_<seq_name>.txt     # Middle product to mark good frame ID 
    +-- good_rad_idx_<seq_name>.txt # Middle product to mark good radar frame ID

+-- calib   # time mapping between sensors
    +-- <sensor_01>2<sensor_02>_mapping.txt
        # Matched unix timestamps from sensor 1 frames and sensor 2 frames
        # Noted: The timestamp between radarpcl (TI radar) and other can be large due to time mismatch between different systems. This has been manually calibrated for each sequence.
        # Format: <sensor_01_ts> <sensor_02_ts>
    +-- ....

+-- gt      # groundtruth as .csv or .json files
    +-- <seq_name>.csv
        # Groundtruth points in Optitrack coordinate
        # Format: <FrameID>,<Timestamp>,<Joint_01_x>,<Joint_01_y>,<Joint_02_z>,...,<Box_Rot_x>,\
        #         <Box_Rot_y>,<Box_Rot_z>,<Box_Rot_w>,<Box_Pos_x>,<Box_Pos_y>,<Box_Pos_z>,\
        #         <Box_Pos_Cov>
    
    +-- <seq_name>_filtered.csv     
        # Groundtruth points in TI radar coordinate
        # Format: <RadFrameID>,<RadTime>,<GroundTruthID>,<GroundTruthTime>,<Joint_01_x>,\
        #         <Joint_01_y>,<Joint_02_z>,...,<Box_Rot_x>,<Box_Rot_y>,<Box_Rot_z>,\
        #         <Box_Rot_w>,<Box_Pos_x>,<Box_Pos_y>,<Box_Pos_z>,<Data_Type>
    
    +-- <seq_name>_filtered_bosch.csv     
        # Groundtruth points in Bosch radar coordinate
        # Format: <RadFrameID>,<RadTime>,<GroundTruthID>,<GroundTruthTime>,<Joint_01_x>,\
        #         <Joint_01_y>,<Joint_02_z>,...,<Box_Rot_x>,<Box_Rot_y>,<Box_Rot_z>,\
        #         <Box_Rot_w>,<Box_Pos_x>,<Box_Pos_y>,<Box_Pos_z>,<Data_Type>

    +-- train.json/test.json/val.json
        # Groundtruth in RT-Pose format
        # Format: {
        #           <seq_name>: {
        #             <Timestamp>: {
        #               "Radar_frameID": <Timestamp>,
        #               "LiDAR val": 0,
        #               "pose": [[[Per01_Joint_01_x,Per01_Joint_01_y,Per01_Joint_01_z],...],
        #                        [[Per02_Joint_01_x,Per02_Joint_01_y,Per02_Joint_01_z],...]]     
        #             },
        #             ...
        #           }
        #         }
    
    +-- <seq_name>_train/test/val.json
        # Json type of ground truth for multiple people case using radar point cloud
        # Format: {
            <Timestamp>: {
                # List size depends on number of people in the frame
                "frame_id": [a, ...] 
                
                # List shape (number_of_people, 15, 3)
                # In which: 15 is 15 joints, 3 is 3D position of the joint
                "pose": [[[person_a_joint_0_x, person_a_joint_0_y, person_a_joint_0_z], ...],
                         [[person_b_joint_0_x, person_b_joint_0_y, person_b_joint_0_z], ...]]
            },
            ...
        }

+-- images  # images from ZED camera (Optional)
    +-- <timestamp>.png
    +-- ....

+-- lidar   # lidar pcl from OS0 (Optional)
    +-- <timestamp>.pcd
    +-- ....
    
+-- radar   # radar pcl from TI radar 
    +-- bin # ADC raw data
        +-- <chip_type>_data.bin
        +-- <chip_type>_idx.bin
        +-- ....
    +-- npy_DZYX_real # 4D DZYX tensor
        +-- <timestamp>.npy
            # 4D radar tensor from TI radar
            # Format: 4D Tensor with shape (Doppler_bins, Z_bins, Y_bins, X_bins)
        +-- ....
    +-- radarpcl # radar pcl frames
        +-- <timestamp>.xyzdi
            # 5D radar point cloud from TI radar
            # Format: <Point_x>,<Point_y>,<Point_z>,<Doppler>,<SignalIntensity>
        +-- ....
    +-- timestamps.txt

+-- radarpcl_mars # data tensor for MARS/PointMLP/RadarHPE_ViT
    +-- featuremap_<seq_name>_<type>.npy
        # MARS type feature map of TI radar point cloud
        # Format: 4D Tensor with shape (n_frames, 12, 12, 5)
        #         with (12, 12, 5) is 144 points, sorted using their x and y values
        #         each point has 5 values (x, y, z, Doppler, Signal_Intensity)

    +-- labels_<seq_name>_<type>.npy
        # MARS type ground truth in TI radar coordinate
        # Format: 3D Tensor with shape (n_frames, 15, 3)
        #         with 15 is the number of joints and 3 is for (x, y, z)

    +-- good_idx_<seq_name>.txt     # Middle product to mark good frame ID 
    +-- good_rad_idx_<seq_name>.txt # Middle product to mark good radar frame ID

+-- radarpcl_multiper # data tensor for MARS/PointMLP/RadarHPE_ViT
    +-- featuremap_<seq_name>_<type>.npy
        # MARS type feature map of TI radar point cloud
        # Format: 4D Tensor with shape (n_frames, 12, 12, 5)
        #         with (12, 12, 5) is 144 points, sorted using their x and y values
        #         each point has 5 values (x, y, z, Doppler, Signal_Intensity)

    +-- labels_<seq_name>_<type>.npy
        # MARS type ground truth in TI radar coordinate
        # Format: 3D Tensor with shape (n_frames, 15, 3)
        #         with 15 is the number of joints and 3 is for (x, y, z)

    +-- good_idx_<seq_name>.txt     # Middle product to mark good frame ID 
    +-- good_rad_idx_<seq_name>.txt # Middle product to mark good radar frame ID

+-- openpose # 2D keypoints getting from OpenPose
    +-- <timestamp>_keypoints.json # 2D HPE dictionary
        # Format:
        # version: OpenPose version
        # people: List of person HPE
        #   person_id: ID of a person in the frame, if used re-id.
        #   pose_keypoints_2d: [x_0, y_0, score_0, ...] # (x,y)-joint position on image, score-estimation confidence.
        # ... : Other keys are not used
```

*Note:*

*1. Mars type radar point cloud data tensor are not available for multiple people sequences and outdoor sequences.*

*2. For outdoor sequences, only radar point cloud is available. To get the 4D radar heatmap, see the [TI radar processing guide](../radar_parsing/README.md).*

*3. Folders with optional tag are empty, only for keeping the data structure. Extract the corresponding streams from the ROS 2 bags when needed.*

*4. Joints order in ground truth: Pelvic, Right Hip, Right Knee, Right Ankle, Left Hip, Left Knee, Left Ankle, Back Top, Head Top, Right Shoulder, Right Elbow, Right Wrist, Left Shoulder, Left Elbow, Left Wrist.*
