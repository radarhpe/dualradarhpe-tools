import numpy as np
import os
import os.path as osp
import argparse
import json
from joints_order import j_order
from data_split import data_info

joint_list = list(j_order.keys())
ROI = [0.5, 8.0, -2.5, 2.5, -np.inf, np.inf]

def is_pelvis_in_roi(gt):
    gt_x, gt_y, gt_z = gt[0,0], gt[0,1], gt[0,2]
    is_gt_x_valid = np.all(gt_x >= ROI[0]) and np.all(gt_x <= ROI[1])
    is_gt_y_valid = np.all(gt_y >= ROI[2]) and np.all(gt_y <= ROI[3])
    is_gt_z_valid = np.all(gt_z >= ROI[4]) and np.all(gt_z <= ROI[5])

    return (is_gt_x_valid and is_gt_y_valid and is_gt_z_valid)

def is_gt_in_roi(gt):
    gt_x, gt_y, gt_z = gt[:,0], gt[:,1], gt[:,2]
    is_gt_x_valid = np.all(gt_x >= ROI[0]) and np.all(gt_x <= ROI[1])
    is_gt_y_valid = np.all(gt_y >= ROI[2]) and np.all(gt_y <= ROI[3])
    is_gt_z_valid = np.all(gt_z >= ROI[4]) and np.all(gt_z <= ROI[5])

    return (is_gt_x_valid and is_gt_y_valid and is_gt_z_valid)

def main():
    parser = argparse.ArgumentParser(
        prog="create_rtpose_train_json.py",
        description="Create RT-POSE'style train.json file from gt"
    )

    parser.add_argument(
        "-d", "--dir",
        type=str,
        help="Path to data dir"
    )

    parser.add_argument(
        "-sd", "--save_dir",
        type=str,
        default="",
        help="Path to save dir"
    )

    parser.add_argument(
        "--sel",
        default="",
        help="Select sequence to process instead of processing all sequences"
    )

    args = parser.parse_args()

    save_dir = args.dir
    if (args.save_dir != ""):
        save_dir = args.save_dir

    seqs = sorted(os.listdir(args.dir))
    ignore_seqs = ["civit_norm_07"]

    select_seq = []
    if (args.sel != ""):
        select_seq = [args.sel]
    if (len(select_seq) != 0):
        seqs = select_seq
    else:
        seqs = sorted(os.listdir(args.dir))

    # seqs = ["civit_easy_01", "civit_norm_01", "civit_norm_03",
    #         "civit_norm_04", "civit_norm_05"]
    total_json = {"train": {}, "val": {}, "test": {}}
    for seq in seqs:
        if (seq in ignore_seqs or seq not in data_info):
            print(f"Sequence {seq} is ignored or not available")
            continue

        # Create gt directory
        os.makedirs(osp.join(save_dir, seq, "gt"), exist_ok=True)

        print(seq)
        data_type = data_info[seq]["type"]
        
        # gt file
        gt_path = osp.join(args.dir, seq, "gt", seq[seq.rfind("/")+1:] + "_filtered.csv")
        with open(gt_path, "r") as f:
            gt_csv = f.read().split("\n")[1:-1] # Remove header and last row
        
        if (data_type == "train"):
            seq_json = {"train": {}, "val": {}}
            frame_json = {"train": {}, "val": {}}
        else:
            seq_json = {"test": {}}
            frame_json = {"test": {}}

        n_per = 1 if "multi" not in seq else 2
        for i, gt_line in enumerate(gt_csv):
            # print(i)
            gt = gt_line.split(",")
            name = gt[1]

            # data_type = "train"
            # if (i > test_idx):
            #     data_type = "test"
            # elif (i > val_idx):
            #     data_type = "val"

            # Data type is marked at the last column of the csv data row
            data_type = gt[-1]
            # # This list keeps ground truth for multiple subjects 
            # notation_list = []
            
            is_frame_valid = True
            # This dict keeps ground truth for each Detection
            object_json = {}
            object_json["Radar_frameID"] = name
            object_json["LiDAR val"] = 0
            object_json["pose"] = []

            for per in range(n_per):
                # if ("" in gt[4+45*per:4+45*(per+1)]):
                #     continue

                # print(per, len(gt[4+48*per:4+48*(per+1)]))
                # print(gt_line)
                lab = np.array(gt[4+45*per:4+45*(per+1)], np.float32).reshape(-1, 3)
                
                # Check if subject pelvis in ROI
                if (is_pelvis_in_roi(lab)):
                    # If pelvis in roi. Check if everything else in roi.
                    # nan is consider as not in roi
                    if (not is_gt_in_roi(lab)):
                        is_frame_valid = False
                        break
                else:
                    continue

                # 0-Pelvis, 1-Right_Hip, 2-Right_Knee, 
                # 3-Right_Ankle, 4-Left_Hip, 5-Left_Knee, 
                # 6-Left_Ankle, 7-Back_Top, 8-Head,
                # 9-Right_Shoulder, 10-Right_Elbow, 11-Right_Wrist
                # 12-Left_Shoulder, 13-Left_Elbow, 14-Left_Wrist
                pose = lab.tolist()
                
                object_json["pose"].append(pose)
                
                # notation_list.append(object_json)
            
            if (len(object_json["pose"]) == 0):
                is_frame_valid = False

            if (not is_frame_valid):
                continue

            # Check occlusion
            # Occlusion happen when pelvis locations are too close
            if (len(object_json["pose"]) > 1):
                P1_pelvis = np.asarray(object_json["pose"][0][0])
                P2_pelvis = np.asarray(object_json["pose"][1][0])
                origin = np.asarray([0,0,0])

                P1O = np.linalg.norm(origin-P1_pelvis)
                P2O = np.linalg.norm(origin-P2_pelvis)
                P1P2 = np.linalg.norm(P2_pelvis-P1_pelvis)
                delta = abs(np.arccos((P1O**2 + P2O**2 - P1P2**2)/(2*P1O*P2O)))

                if (delta <= 3*np.pi/180):
                    if (P1_pelvis[0] < P2_pelvis[0]):
                        del object_json["pose"][1]
                    else:
                        del object_json["pose"][0]

            frame_json[data_type][name] = object_json
            seq_json[data_type][seq] = frame_json[data_type]

        # Save ground truth json for each seq
        for data_type in ["train", "val", "test"]:
            if (data_type in seq_json):
                with open(osp.join(save_dir, seq, f"gt/{data_type}.json"), "w") as f:
                    json.dump(seq_json[data_type], f, indent=2)
                
                # Put seq ground truth to total ground truth
                total_json[data_type] = {**total_json[data_type], **seq_json[data_type]}

    # Save ground truth json for all seq
    os.makedirs(osp.join(save_dir, "../rtpose_anno"), exist_ok=True)
    for data_type in ["train", "val", "test"]:
        with open(osp.join(save_dir, "../rtpose_anno", f"{data_type}.json"), "w") as f:
            json.dump(total_json[data_type], f, indent=2)
            
        # break
if __name__ == "__main__":
    main()