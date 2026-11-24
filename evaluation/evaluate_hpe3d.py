import numpy as np
import os
import os.path as osp
import json
from scipy.spatial.transform import Rotation
import argparse

j_order = ["Pelvic", "WaistRBack", "RKneeOut", "RAnkleOut", "WaistLBack", "LKneeOut", 
             "LAnkleOut", "BackTop", "HeadTop", "RShoulderTop","RElbowOut", 
             "RWristOut", "LShoulderTop", "LElbowOut", "LWristOut"]

left_arm_idx = [4, 12, 13]
right_arm_idx = [1, 9, 10]
left_leg_idx = [12, 4, 5]
right_leg_idx = [9, 1, 2]

left_akl_idx = [12, 13, 14]
right_akl_idx = [9, 10, 11]
left_kne_idx = [4, 5, 6]
right_kne_idx = [1, 2, 3]

def AlignMPJPE(pred, gt):
    pred -= pred[:,0:1,:]
    gt -= gt[:,0:1,:]

    return MPJPE(pred, gt)

def MPJPE(pred, gt):
    # input shape: (batch_size, 15, 3)
    mpjpe_j = np.linalg.norm(pred - gt, axis=2)
    mean_mpjpe = np.mean(mpjpe_j)
    mean_mpjpe = np.append(mean_mpjpe, np.mean(mpjpe_j, axis=0))
    std_mpjpe = np.std(mpjpe_j)
    std_mpjpe = np.append(std_mpjpe, np.std(mpjpe_j, axis=0))
    return mean_mpjpe, std_mpjpe

def AngleErr(pred, gt):
    def cal_joint_angle(joints):
        # Get dot product => cosine angle
        P1, P2, P3 = joints[:,0,:], joints[:,1,:], joints[:,2,:]
        P21 = P1 - P2
        P23 = P3 - P2
        
        dot_product = np.einsum('ij,ij->i', P21, P23)
        cosine_angle = dot_product / (np.linalg.norm(P21, axis=1)*np.linalg.norm(P23, axis=1) + 1e-10)
        # cosine_angle[np.abs(cosine_angle) > 1.] = 1.
        angle = np.arccos(cosine_angle)
        # print(angle)
        return np.expand_dims(angle, axis=1)

    def cal_batch_angle(batch):
        return np.hstack((cal_joint_angle(batch[:,left_arm_idx,:]),
                        cal_joint_angle(batch[:,right_arm_idx,:]),
                        cal_joint_angle(batch[:,left_leg_idx,:]),
                        cal_joint_angle(batch[:,right_leg_idx,:]),
                        cal_joint_angle(batch[:,left_akl_idx,:]),
                        cal_joint_angle(batch[:,right_akl_idx,:]),
                        cal_joint_angle(batch[:,left_kne_idx,:]),
                        cal_joint_angle(batch[:,right_kne_idx,:])))

    pred_angles = cal_batch_angle(pred)
    gt_angles = cal_batch_angle(gt)

    # angles_diff = ((pred_angles - gt_angles) + np.pi) % (2*np.pi) - np.pi
    angles_diff = pred_angles - gt_angles
    mean_angles_score = np.mean(np.abs(angles_diff))
    mean_angles_score = np.append(mean_angles_score, np.mean(np.abs(angles_diff), axis=0))
    std_angles_score = np.std(np.abs(angles_diff))
    std_angles_score = np.append(std_angles_score, np.std(np.abs(angles_diff), axis=0))

    return mean_angles_score, std_angles_score

def main():
    parser = argparse.ArgumentParser(
        prog="evaluate_hpe3d.py",
        description="Evaluate estimation of the 3D HPE"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="PointMLP",
        help="Model type"
    )

    parser.add_argument(
        "--date",
        type=str,
        default="",
        help="Date record"
    )

    parser.add_argument(
        "--use_motion",
        action="store_true",
    )

    parser.add_argument(
        "--prefix",
        default="",
        help="Define folder prefix"
    )
    
    parser.add_argument(
        "--suffix",
        default="",
        help="Define test suffix"
    )

    parser.add_argument(
        "--motion_only",
        action="store_true",
    )

    args = parser.parse_args()

    suffix = f"_{args.suffix}" if args.suffix != "" else "" 
    if (args.model != "RT_Pose"):
        if (args.use_motion):
            gt_path = f"./groundtruth/{args.date}/{args.prefix}/with_motion/labels_test{suffix}.npy"
            pred_path = f"./{args.model}/{args.date}/{args.prefix}/with_motion/infer_out{suffix}.npy"
        else:
            gt_path = f"./groundtruth/{args.date}/{args.prefix}/no_motion/labels_test{suffix}.npy"
            pred_path = f"./{args.model}/{args.date}/{args.prefix}/no_motion/infer_out{suffix}.npy"

        # Load prediction and ground truth
        pred = np.load(pred_path)
        if (len(pred.shape) != 3):
            pred = pred.reshape(-1, 15, 3)
        gt = np.load(gt_path)
        if (len(gt.shape) != 3):
            gt = gt.reshape(-1, 15, 3)

        # pred = pred[1647:2803]
        # gt = gt[1647:2803]
    else:
        gt_path = f"./groundtruth/test.json"
        if (args.use_motion):
            pred_path = f"./{args.model}/{args.date}/{args.prefix}/with_motion/test_prediction{suffix}.json"
        elif (not args.motion_only):
            pred_path = f"./{args.model}/{args.date}/{args.prefix}/no_motion/test_prediction{suffix}.json"
        else:
            pred_path = f"./{args.model}/{args.date}/{args.prefix}/with_motion/test_prediction_mo-only.json"

        with open(gt_path, "r") as f:
            gt_json = json.load(f)
        with open(pred_path, "r") as f:
            pred_json = json.load(f)

        if (args.motion_only):
            gt_json = {key: gt_json[key] for key in list(pred_json.keys())}

        pred, gt = np.empty((0, 15, 3)), np.empty((0, 15, 3))
        for seq, frame in pred_json.items():
            for k, v in frame.items():
                kps = np.asarray(v["keypoints"])[:,:,1:4]
                pred = np.append(pred, kps, axis=0)
            # TODO: This is processing for one person only
            for k, v in gt_json[seq].items():
                kps = np.asarray(v["pose"][0:1])
                gt = np.append(gt, kps, axis=0)
    
    # 
    print(pred.shape, gt.shape) 
    mean_mpjpe, std_mpjpe = MPJPE(pred, gt)
    mean_al_mpjpe, std_al_mpjpe = AlignMPJPE(pred, gt)

    #
    mean_angle_err, std_angle_err = AngleErr(pred, gt)

    # print(mpjpe)
    # print(angle_err)

    output_file = f"{args.model}_{args.date}{suffix}_eval.json"
    if (args.motion_only):
        output_file = f"{args.model}_{args.date}_hard_eval.json"
    output_path = pred_path[:pred_path.rfind("/")+1] + output_file
    result_json = {
        "Model_name": args.model,
        "Date": args.date,
        "MPJPE": dict([("MPJPE_all", (mean_mpjpe[0], std_mpjpe[0]))] + \
                        [("MPJPE_"+j_order[i]+str(i), (mean_mpjpe[i+1], std_mpjpe[i+1])) 
                        for i in range(mean_mpjpe.shape[0]-1)]),
        "Align_MPJPE": dict([("Align_MPJPE_all", (mean_al_mpjpe[0], std_al_mpjpe[0]))] + \
                        [("Align_MPJPE_"+j_order[i]+str(i), (mean_al_mpjpe[i+1], std_al_mpjpe[i+1])) 
                        for i in range(mean_al_mpjpe.shape[0]-1)]),
        "AngleErr": {
            "AngErr_all": (mean_angle_err[0], std_angle_err[0]),
            "AngErr_right_elbow": (mean_angle_err[6], std_angle_err[6]), 
            "AngErr_right_arm": (mean_angle_err[2], std_angle_err[2]), 
            "AngErr_right_leg": (mean_angle_err[4], std_angle_err[4]), 
            "AngErr_right_knee": (mean_angle_err[8], std_angle_err[8]), 
            "AngErr_left_elbow": (mean_angle_err[5], std_angle_err[5]), 
            "AngErr_left_arm": (mean_angle_err[1], std_angle_err[1]), 
            "AngErr_left_leg": (mean_angle_err[3], std_angle_err[3]), 
            "AngErr_left_knee": (mean_angle_err[7], std_angle_err[7]), 
        }
    }
    with open(output_path, "w") as f:
        json.dump(result_json, f, indent=2)

if __name__ == "__main__":
    main()