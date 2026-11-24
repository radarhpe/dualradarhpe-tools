import numpy as np
import os
import os.path as osp
import json
from scipy.spatial.transform import Rotation
from scipy.spatial import distance_matrix, KDTree
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

def AlignMPJPE(pred, gt, match_pairs, real_n_person):
    pred -= pred[:,:,0:1,:]
    gt -= gt[:,:,0:1,:]

    return MPJPE(pred, gt, match_pairs, real_n_person)

def MPJPE(pred, gt, match_pairs, real_n_person):
    # Shape: (n_frames, 2, 1)
    valid_idx = match_pairs[:,:,0] != -1

    # input shape: (n_frames, 2, 15, 3) -> (n_frames, 2, 15)
    mpjpe_j = np.linalg.norm(pred - gt, axis=3)

    # reduce (n_frames, 2, 15) to (n_frames, 15)
    mpjpe_j = mpjpe_j[valid_idx]
    # calculate final metrics
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
        cosine_angle = dot_product / (np.linalg.norm(P21, axis=1)*np.linalg.norm(P23, axis=1))
        angle = np.arccos(cosine_angle)
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

    angles_diff = ((pred_angles - gt_angles) + np.pi) % (2*np.pi) - np.pi
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

    gt_path = f"./groundtruth/test.json"
    pred_path = f"./RT_Pose/01-05-2025/multi_person/test_prediction.json"

    with open(gt_path, "r") as f:
        gt_json = json.load(f)
    with open(pred_path, "r") as f:
        pred_json = json.load(f)

    pred, gt = np.empty((0, 2, 15, 3)), np.empty((0, 2, 15, 3))
    real_n_person = []
    match_pairs = []
    for seq, frame in pred_json.items():
        for k, v in frame.items():
            # Keypoints shape (num_of_person, 15, 3)
            kps = np.asarray(v["keypoints"])[:,:,1:4]
            pred = np.append(pred, np.expand_dims(kps, axis=0), axis=0)

            # Get same frame in gt
            gt_kps = np.asarray(gt_json[seq][k[k.rfind("_")+1:]]["pose"])
            real_n_person.append(gt_kps.shape[0])

            # Calculate distance matrix to match pred and gt
            dist_mat = distance_matrix(gt_kps[:,0,:], kps[:,0,:])
            is_min = np.min(dist_mat, axis=1) == dist_mat
            is_min_idx = np.transpose(np.nonzero(is_min))
            match_pair = -1 * np.ones((2,2))
            match_pair[:is_min_idx.shape[0], :] = is_min_idx
            match_pairs.append(match_pair)

            # Pad zero if gt only has one person
            if (gt_kps.shape[0] != 2):
                gt_kps = np.append(gt_kps, np.zeros((1, 15, 3)), axis=0)
            gt = np.append(gt, np.expand_dims(gt_kps, axis=0), axis=0)
    # 
    match_pairs = np.array(match_pairs)
    # print(pred.shape, gt.shape, match_pairs) 
    mean_mpjpe, std_mpjpe = MPJPE(pred, gt, match_pairs, real_n_person)
    mean_al_mpjpe, std_al_mpjpe = AlignMPJPE(pred, gt, match_pairs, real_n_person)

    #
    # mean_angle_err, std_angle_err = AngleErr(pred, gt)

    # print(mpjpe)
    # print(angle_err)

    output_file = f"RT_Pose_01-05-2025_multiple_person_eval.json"
    output_path = pred_path[:pred_path.rfind("/")+1] + output_file
    result_json = {
        "Model_name": "RT_Pose",
        "Date": "01-05-2025",
        "MPJPE": dict([("MPJPE_all", (mean_mpjpe[0], std_mpjpe[0]))] + \
                        [("MPJPE_"+j_order[i]+str(i), (mean_mpjpe[i+1], std_mpjpe[i+1])) 
                        for i in range(mean_mpjpe.shape[0]-1)]),
        "Align_MPJPE": dict([("Align_MPJPE_all", (mean_al_mpjpe[0], std_al_mpjpe[0]))] + \
                        [("Align_MPJPE_"+j_order[i]+str(i), (mean_al_mpjpe[i+1], std_al_mpjpe[i+1])) 
                        for i in range(mean_al_mpjpe.shape[0]-1)]),
        # "AngleErr": {
        #     "AngErr_all": (mean_angle_err[0], std_angle_err[0]),
        #     "AngErr_right_elbow": (mean_angle_err[6], std_angle_err[6]), 
        #     "AngErr_right_arm": (mean_angle_err[2], std_angle_err[2]), 
        #     "AngErr_right_leg": (mean_angle_err[4], std_angle_err[4]), 
        #     "AngErr_right_knee": (mean_angle_err[8], std_angle_err[8]), 
        #     "AngErr_left_elbow": (mean_angle_err[5], std_angle_err[5]), 
        #     "AngErr_left_arm": (mean_angle_err[1], std_angle_err[1]), 
        #     "AngErr_left_leg": (mean_angle_err[3], std_angle_err[3]), 
        #     "AngErr_left_knee": (mean_angle_err[7], std_angle_err[7]), 
        # }
    }
    with open(output_path, "w") as f:
        json.dump(result_json, f, indent=2)

if __name__ == "__main__":
    main()