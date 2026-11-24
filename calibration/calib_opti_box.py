import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import os.path as osp
import argparse
import time
import json
from scipy.spatial.transform import Rotation

from extrinsic_calib import T_box_lidar, T_lidar_ti, create_transformation_matrix, \
                            src2dst_transformation

def normalize(v):
    return v / np.linalg.norm(v)

def rotate_vector(points):
    """
    Rotates vector v1 to align with vector v2 using Rodrigues' rotation formula.
    
    Args:
        v1: The initial vector (numpy array)
        v2: The target vector to align with (numpy array)
        
    Returns:
        rotated_v: The rotated vector (numpy array)
    """
    # Step 1: get norm vector
    A, B, C = points
    AB = B - A
    AC = C - A
    v1 = np.array([1, 0, 0])
    v2 = normalize(np.cross(AB, AC))
    
    # Step 2: Compute the axis of rotation (cross product of v1 and v2)
    axis = np.cross(v1, v2)
    
    # If the vectors are already aligned, return v1
    if np.linalg.norm(axis) < 1e-6:
        return v1
    
    # Normalize the axis
    axis = normalize(axis)
    
    # Step 3: Compute the angle of rotation (dot product between v1 and v2)
    angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
    
    # Step 4: Rodrigues' rotation formula
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    # Rotation matrix (3x3)
    R = np.array([
        [cos_angle + axis[0]**2 * (1 - cos_angle), axis[0] * axis[1] * (1 - cos_angle) - axis[2] * sin_angle, axis[0] * axis[2] * (1 - cos_angle) + axis[1] * sin_angle],
        [axis[1] * axis[0] * (1 - cos_angle) + axis[2] * sin_angle, cos_angle + axis[1]**2 * (1 - cos_angle), axis[1] * axis[2] * (1 - cos_angle) - axis[0] * sin_angle],
        [axis[2] * axis[0] * (1 - cos_angle) - axis[1] * sin_angle, axis[2] * axis[1] * (1 - cos_angle) + axis[0] * sin_angle, cos_angle + axis[2]**2 * (1 - cos_angle)]
    ])
    return R

def rotmat_to_euler(R):
    return Rotation.from_rotvec(R).as_euler('zyx', degrees=True)

def euler_to_rotmat(R):
    pass

def rotate_point(points, R):
    # 4x4
    R_mat = np.eye(4)
    R_mat[:3,:3] = R
    # print(R_mat)
    
    # 4x3
    p_mat = np.ones((3, 4))
    p_mat[:,:3] = points
    p_mat = p_mat.T
    # print(p_mat)

    # 4x3
    p_rot = np.matmul(R_mat, p_mat)
    return p_rot[:3,:].T

def main():
    parser = argparse.ArgumentParser(
        prog="visualize_points.py",
        description="For calculating centroid of reflector on lidar"
    )

    # Box to ideal TI
    R = np.array([[0, -1, 0],
                  [0,  0, 1],
                  [-1, 0, 0]])

    # Point in OptiTrack coord
    points_06_03 = {"points": np.array([[-0.292822,0.77576,3.216472],
                        [-0.471563,0.776792,3.213037],
                        [-0.377317,1.085604,3.20402]]), 
                    "date": "march",
                    "box_loc": np.array([0.000437, 0.00027, 0.000203, -1, 
                                         -0.369669, 1.926317, 3.281397])}

    points_01_04 = {"points": np.array([[-0.073997,1.236251,4.194365],
                        [0.105793,1.230353,4.182975],
                        [0.007288,0.926631,4.198318]]), 
                    "date": "april",
                    "box_loc": np.array([0.001617, 0.010839, 0.000257, -0.99994, 
                                         0.049132, 2.075172, 4.267977])}

    points_08_04 = {"points": np.array([[-0.56116, 1.214603, 2.581806],
                        [-0.382164,1.204216,2.60049],
                        [-0.489338,0.902985,2.613346]]),
                    "date": "april2",
                    "box_loc": np.array([2.8E-05, 5.2E-05, 2.7E-05, -1, 
                                         -0.429927, 2.052438, 2.638323])}

    points_22_04 = {"points": np.array([[-0.657309, 1.203527, 2.709681],
                        [-0.477413, 1.196279, 2.721558],
                        [-0.579017, 0.892944, 2.733797]]),
                    "date": "april3",
                    "box_loc": np.array([-0.000106, -3.7E-05, -0.000198, -1, 
                                         -0.53783, 2.042829, 2.77238])}

    points_30_04 = {"points": np.array([[-0.36378, 1.305288, 2.936528],
                                       [-0.183538, 1.297181, 2.938368],
                                       [-0.28581, 0.99504, 2.965547]]),
                    "date": "april4",
                    "box_loc": np.array([0.014954, -0.028538, 0.002762, -0.999477, 
                                         -0.236681, 2.145262, 2.966885])}

    # Change this to match each set
    p = points_30_04
    points, date, box_loc = p["points"], p["date"], p["box_loc"]
    T_opti_box = create_transformation_matrix(box_loc[:4], box_loc[4:])

    # Transform OpiTrack point to Box coord
    points_box = src2dst_transformation(np.linalg.inv(T_opti_box), points)

    # Transform Box to lidar coord
    points_li = src2dst_transformation(np.linalg.inv(T_box_lidar[date]), points_box)

    # Transform lidar points to TI coord
    points_ti = src2dst_transformation(np.linalg.inv(T_lidar_ti), points_li)

    # 

    # Rotate point in TI to align with ground (x axis is parallel with ground)
    R_mat = rotate_vector(points_ti)
    print(R_mat)
    
    # Rotate points to ideal TI
    points_rot = (R_mat.T @ points_ti.T).T
    print("points: ", points_rot)

    # Get ideal TI to real TI rotation
    roll_angle = np.arctan2(points_rot[0,2]-points_rot[1,2],
                            points_rot[0,1]-points_rot[1,1])
    roll_angle = (roll_angle - np.pi) if (roll_angle >= np.pi/2) else roll_angle
    roll_angle = (roll_angle + np.pi) % (2*np.pi) - np.pi
    print(roll_angle)

    R_roll = np.array([[1, 0, 0],
                       [0, np.cos(roll_angle), -np.sin(roll_angle)],
                       [0, np.sin(roll_angle), np.cos(roll_angle)]])

    print((R_roll.T @ points_rot.T).T)
    R_calib_ti = Rotation.from_matrix(np.matmul(R_mat, R_roll))
    print("Rotation from real TI to calib TI: \nas quat: ", R_calib_ti.as_quat(), 
                                      "\nas matrix", R_calib_ti.as_matrix())
    
    # Get rotation from box to lidar
    # R_lidar_ti = Rotation.from_quat([0.00780112, 0.00873333, 0.21170263, 0.97726397]).as_matrix()
    # R_box_lidar = np.matmul(R_box_ti.as_matrix(), R_lidar_ti.T)
    # print("Rotation from box to lidar: \n", Rotation.from_matrix(R_box_lidar).as_quat(), "\n",
    #       R_box_lidar)
    
    # adjust = 
    # R_adjust = np.array([[np.cos(roll_angle), -np.sin(roll_angle), 0],
    #                    [np.sin(roll_angle), np.cos(roll_angle), 0],
    #                    [0, 0, 1]])

    # # 
    # fig = plt.figure(figsize=(12,10))
    # fig.tight_layout()

    # ax11 = fig.add_subplot(111, projection="3d")
    # ax11.view_init(elev=10, azim=-165)

    # # line_idx = np.array([[5,4],[4,9],[9,10],[10,11],[4,6],[6,7],[7,8],
    # #                      [9,1],[6,0],[1,0],[1,14],[14,15],[0,12],[12,13]])

    # ax11.cla()
    # ax11.scatter(points[:,0], points[:,1], points[:,2], s=5, c="red")
    # scale = 1.0
    # ax11.quiver(*point, *X*scale, color='r', label='X axis')
    # ax11.quiver(*point, *Y*scale, color='g', label='Y axis')
    # ax11.quiver(*point, *Z*scale, color='b', label='Z axis (normal)')

    
    # # ax11.scatter(gt[:,0], gt[:,1], gt[:,2], s=5, c='blue')
    # # for id in line_idx:
    # #     ax11.plot([gt[id[0],0], gt[id[1],0]],
    # #                 [gt[id[0],1], gt[id[1],1]],
    # #                 [gt[id[0],2], gt[id[1],2]], c='blue')

    # ax11.set_xlabel('X-metre')
    # ax11.set_ylabel('Y-metre')
    # ax11.set_zlabel('Z-metre')

    # ax11.set_xlim(-2, 2)
    # ax11.set_ylim(0, 3)
    # ax11.set_zlim(2.0, 4.5)

    # plt.show()

if __name__ == "__main__":
    main()
